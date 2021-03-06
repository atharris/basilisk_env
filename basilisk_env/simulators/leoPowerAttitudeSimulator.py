# 3rd party modules
import numpy as np

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.simulation import sim_model

from Basilisk.simulation import (spacecraftPlus, gravityEffector, extForceTorque, simple_nav, spice_interface,
                                eclipse, imu_sensor, exponentialAtmosphere, facetDragDynamicEffector, planetEphemeris, reactionWheelStateEffector,
                                 thrusterDynamicEffector)
from Basilisk.simulation import simMessages
from Basilisk.utilities import simIncludeRW, simIncludeGravBody, simIncludeThruster
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import simulationArchTypes
from Basilisk.simulation import simpleBattery, simplePowerSink, simpleSolarPanel
from Basilisk import __path__
bskPath = __path__[0]

from Basilisk.fswAlgorithms import inertial3D, hillPoint, celestialTwoBodyPoint
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import MRP_Feedback, rwMotorTorque
from Basilisk.fswAlgorithms import fswMessages
from Basilisk.fswAlgorithms import inertialUKF
from Basilisk.fswAlgorithms import thrMomentumManagement, thrMomentumDumping, thrForceMapping, thrFiringSchmitt

from basilisk_env.simulators.dynamics.effectorPrimatives import actuatorPrimatives as ap
from basilisk_env.simulators.initial_conditions import leo_orbit, sc_attitudes
from numpy.random import uniform

class LEOPowerAttitudeSimulator(SimulationBaseClass.SimBaseClass):
    '''
    Simulates a spacecraft in LEO with atmospheric drag and J2.

    Dynamics Components
    - Forces: J2, Atmospheric Drag w/ COM offset
    - Environment: Exponential density model; eclipse
    - Actuators: ExternalForceTorque
    - Sensors: SimpleNav
    - Systems: SimpleBattery, SimpleSink, SimpleSolarPanel

    FSW Components:
    - MRP_Feedback controller
    - inertial3d (sun pointing), hillPoint (nadir pointing)
    - [WIP, not implemented] inertialEKF attitude filter

    Action Space (discrete, 0 or 1):
    0 - orients the spacecraft towards the sun.
    1 - orients the spacecraft towards Earth.
    [WIP, not implemented] 2 - Runs inertialEKF to better estimate the spacecraft state.

    Observation Space:
    eclipseIndicator - float [0,1] - indicates the flux mitigator due to eclipse.
    storedCharge - float [0,batCapacity] - indicates the s/c battery charge level in W-s.
    |sigma_RB| - float [0,1] - norm of the spacecraft error MRP with respect to the last reference frame specified.
    |diag(filter_covar)| - float - norm of the diagonal of the EKF covariance matrix at that time.

    Reward Function:
    r = 1/(1+ | sigma_RB|) if action=1
    Intended to provide a rich reward in action 1 when the spacecraft is pointed towards the earth, decaying as sigma^2
    as the pointing error increases.

    :return:
    '''
    def __init__(self, dynRate, fswRate, step_duration, initial_conditions=None):
        '''
        Creates the simulation, but does not initialize the initial conditions.
        '''
        self.dynRate = dynRate
        self.fswRate = fswRate
        self.step_duration = step_duration

        SimulationBaseClass.SimBaseClass.__init__(self)
        self.TotalSim.terminateSimulation()

        self.simTime = 0.0

        # If no initial conditions are defined yet, set ICs
        if initial_conditions == None:
            self.initial_conditions=self.set_ICs()
        # If ICs were passed through, use the ones that were passed through
        else:
            self.initial_conditions = initial_conditions

        #   Specify some simulation parameters
        self.mass = self.initial_conditions.get("mass") #kg
        self.powerDraw = self.initial_conditions.get("powerDraw") #W

        self.DynModels = []
        self.FSWModels = []

        #   Initialize the dynamics+fsw task groups, modules

        self.DynamicsProcessName = 'DynamicsProcess' #Create simulation process name
        self.dynProc = self.CreateNewProcess(self.DynamicsProcessName) #Create process
        self.dynTaskName = 'DynTask'
        self.spiceTaskName = 'SpiceTask'
        self.envTaskName = 'EnvTask'
        self.dynTask = self.dynProc.addTask(self.CreateNewTask(self.dynTaskName, mc.sec2nano(self.dynRate)))
        self.spiceTask = self.dynProc.addTask(self.CreateNewTask(self.spiceTaskName, mc.sec2nano(self.step_duration)))
        self.envTask = self.dynProc.addTask(self.CreateNewTask(self.envTaskName, mc.sec2nano(self.dynRate)))

        self.obs = np.zeros([5,1])
        self.sim_states = np.zeros([11,1])

        self.set_dynamics()
        self.set_fsw()

        self.set_logging()
        self.previousPointingGoal = "sunPointTask"

        self.modeRequest = None
        self.InitializeSimulationAndDiscover()

        return

    def set_ICs(self):
        # Sample orbital parameters
        oe, rN,vN = leo_orbit.sampled_400km()

        # Sample attitude and rates
        sigma_init, omega_init = sc_attitudes.random_tumble(maxSpinRate=0.00001)

        # Dict of initial conditions
        initial_conditions = {
            # Mass
            "mass": 330, # kg

            # Orbital parameters
            "oe": oe,
            "rN": rN,
            "vN": vN,

            # Spacecraft dimensions
            "width": 1.38,
            "depth": 1.04,
            "height": 1.58,

            # Attitude and rate initialization
            "sigma_init": sigma_init,
            "omega_init": omega_init,

            # Atmospheric density
            "planetRadius": orbitalMotion.REQ_EARTH * 1000.,
            "baseDensity": 1.22, #kg/m^3
            "scaleHeight": 8e3,  #m

            # Disturbance Torque
            "disturbance_magnitude": 2e-4,
            "disturbance_vector": np.random.standard_normal(3),

            # Reaction Wheel speeds
            "wheelSpeeds": uniform(-800,800,3), # RPM

            # Solar Panel Parameters
            "nHat_B": np.array([0,-1,0]),
            "panelArea": 0.2*0.3,
            "panelEfficiency": 0.20,

            # Power Sink Parameters
            "powerDraw": -5.0, # W

            # Battery Parameters
            "storageCapacity": 20.0 * 3600.,
            "storedCharge_Init": np.random.uniform(8.*3600., 20.*3600., 1)[0],

            # Sun pointing FSW config
            "sigma_R0N": [1,0,0],

            # RW motor torque and thruster force mapping FSW config
            "controlAxes_B": [1, 0, 0,
                             0, 1, 0,
                             0, 0, 1],

            # Attitude controller FSW config
            "K": 7,
            "Ki": -1.0,  # Note: make value negative to turn off integral feedback
            "P": 35,

            # Momentum dumping config
            "hs_min": 4.,  # Nms

            # Thruster force mapping FSW module
            "thrForceSign": 1,

            # Thruster momentum dumping FSW config
            "maxCounterValue": 4,
            "thrMinFireTime": 0.002 #   Seconds
        }

        return initial_conditions

    def set_dynamics(self):
        '''
        Sets up the dynamics modules for the sim. This simulator runs:
        scObject (spacecraft dynamics simulation)
        SpiceObject
        EclipseObject (simulates eclipse for simpleSolarPanel)
        extForceTorque (attitude actuation)
        simpleNav (attitude determination/sensing)
        simpleSolarPanel (attitude-dependent power generation)
        simpleBattery (power storage)
        simplePowerNode (constant power draw)

        By default, parameters are set to those for a 6U cubesat.
        :return:
        '''
        sc_number=0

        #   Spacecraft, Planet Setup
        self.scObject = spacecraftPlus.SpacecraftPlus()
        self.scObject.ModelTag = 'spacecraft'

        # clear prior gravitational body and SPICE setup definitions
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        # setup Spice interface for some solar system bodies
        timeInitString = '2021 MAY 04 07:47:48.965 (UTC)'
        self.gravFactory.createSpiceInterface(bskPath + '/supportData/EphemerisData/'
                                              , timeInitString
                                              , spicePlanetNames=["sun", "earth"]
                                              )

        self.gravFactory.spiceObject.zeroBase="earth" # Make sure that the Earth is the zero base

        self.gravFactory.createSun()
        planet = self.gravFactory.createEarth()
        planet.isCentralBody = True  # ensure this is the central gravitational body
        mu = planet.mu
        # attach gravity model to spaceCraftPlus
        self.scObject.gravField.gravBodies = spacecraftPlus.GravBodyVector(list(self.gravFactory.gravBodies.values()))

        oe = self.initial_conditions.get("oe")
        rN = self.initial_conditions.get("rN")
        vN = self.initial_conditions.get("vN")

        n = np.sqrt(mu / oe.a / oe.a / oe.a)
        P = 2. * np.pi / n

        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")

        I = [1./12.*self.mass*(width**2. + depth**2.), 0., 0.,
             0., 1./12.*self.mass*(depth**2. + height**2.), 0.,
             0., 0.,1./12.*self.mass*(width**2. + height**2.)]

        self.scObject.hub.mHub = self.mass # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        sigma_init = self.initial_conditions.get("sigma_init")
        omega_init = self.initial_conditions.get("omega_init")

        self.scObject.hub.sigma_BNInit = sigma_init  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = omega_init

        self.sim_states[0:3,0] = np.asarray(self.scObject.hub.sigma_BNInit).flatten()
        self.sim_states[3:6,0] = np.asarray(self.scObject.hub.r_CN_NInit).flatten()
        self.sim_states[6:9,0] = np.asarray(self.scObject.hub.v_CN_NInit).flatten()

        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsgName)
        self.densityModel.planetRadius = self.initial_conditions.get("planetRadius")
        self.densityModel.baseDensity = self.initial_conditions.get("baseDensity")
        self.densityModel.scaleHeight = self.initial_conditions.get("scaleHeight")

        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        #   Set up the goemetry of a 6U cubesat
        self.dragEffector.addFacet(0.2*0.3, 2.2, [1,0,0], [0.05, 0.0, 0])
        self.dragEffector.addFacet(0.2*0.3, 2.2, [-1,0,0], [0.05, 0.0, 0])
        self.dragEffector.addFacet(0.1*0.2, 2.2, [0,1,0], [0, 0.15, 0])
        self.dragEffector.addFacet(0.1*0.2, 2.2, [0,-1,0], [0, -0.15, 0])
        self.dragEffector.addFacet(0.1*0.3, 2.2, [0,0,1], [0,0, 0.1])
        self.dragEffector.addFacet(0.1*0.3, 2.2, [0,0,-1], [0, 0, -0.1])
        self.dragEffector.addFacet(1.*2., 2.2, [0,1,0],[0,2.,0])
        self.dragEffector.addFacet(1.*2., 2.2, [0,-1,0],[0,2.,0])

        self.dragEffector.atmoDensInMsgName = self.densityModel.envOutMsgNames[-1]
        self.scObject.addDynamicEffector(self.dragEffector)

        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPositionMsgName(self.scObject.scStateOutMsgName)
        self.eclipseObject.addPlanetName('earth')

        #   Disturbance Torque Setup
        disturbance_magnitude = self.initial_conditions.get("disturbance_magnitude")
        disturbance_vector = self.initial_conditions.get("disturbance_vector")
        unit_disturbance = disturbance_vector/np.linalg.norm(disturbance_vector)
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.extTorquePntB_B = disturbance_magnitude * disturbance_vector
        self.extForceTorqueObject.ModelTag = 'DisturbanceTorque'
        self.extForceTorqueObject.cmdForceInertialInMsgName = 'disturbance_torque'
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

        # Add reaction wheels to the spacecraft
        self.rwStateEffector, rwFactory, initWheelSpeeds = ap.balancedHR16Triad(useRandom=True,randomBounds=(-800,800))
        # Change the wheel speeds
        rwFactory.rwList["RW1"].Omega = self.initial_conditions.get("wheelSpeeds")[0]*mc.RPM #rad/s
        rwFactory.rwList["RW2"].Omega = self.initial_conditions.get("wheelSpeeds")[1]*mc.RPM #rad/s
        rwFactory.rwList["RW3"].Omega = self.initial_conditions.get("wheelSpeeds")[2]*mc.RPM #rad/s
        initWheelSpeeds = self.initial_conditions.get("wheelSpeeds")
        self.rwStateEffector.InputCmds = "rwTorqueCommand"
        rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.rwConfigMsgName = "rwConfig"
        unitTestSupport.setMessage(self.TotalSim, self.DynamicsProcessName, self.rwConfigMsgName, rwFactory.getConfigMessage(), msgStrName="RWArrayConfigFswMsg")

        #   Add thrusters to the spacecraft
        self.thrusterSet, thrFactory = ap.idealMonarc1Octet()
        self.thrusterSet.InputCmds = "rwDesatTimeOnCmd"
        thrModelTag = "ACSThrusterDynamics"
        self.thrusterConfigMsgName = "thrusterConfig"
        unitTestSupport.setMessage(self.TotalSim, self.DynamicsProcessName, self.thrusterConfigMsgName, thrFactory.getConfigMessage(), msgStrName="THRArrayConfigFswMsg")
        thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        #   Add simpleNav as a mock estimator to the spacecraft
        self.simpleNavObject = simple_nav.SimpleNav()
        self.simpleNavObject.ModelTag = 'SimpleNav'
        self.simpleNavObject.inputStateName = self.scObject.scStateOutMsgName

        #   Power Setup
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = 'solarPanel' + str(sc_number)
        self.solarPanel.stateInMsgName = self.scObject.scStateOutMsgName
        self.solarPanel.sunEclipseInMsgName = 'eclipse_data_'+str(sc_number)
        self.solarPanel.sunInMsgName = 'sun_planet_data'
        self.solarPanel.setPanelParameters(unitTestSupport.np2EigenVectorXd(self.initial_conditions.get("nHat_B")), self.initial_conditions.get("panelArea"), self.initial_conditions.get("panelEfficiency"))
        self.solarPanel.nodePowerOutMsgName = "panelPowerMsg" + str(sc_number)

        self.powerSink = simplePowerSink.SimplePowerSink()
        self.powerSink.ModelTag = "powerSink" + str(sc_number)
        self.powerSink.nodePowerOut = self.powerDraw  # Watts
        self.powerSink.nodePowerOutMsgName = "powerSinkMsg" + str(sc_number)

        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.batPowerOutMsgName = "powerMonitorMsg"
        self.powerMonitor.storageCapacity = self.initial_conditions.get("storageCapacity")
        self.powerMonitor.storedCharge_Init = self.initial_conditions.get("storedCharge_Init")
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsgName)
        self.powerMonitor.addPowerNodeToModel(self.powerSink.nodePowerOutMsgName)

        self.sim_states[9,0] = self.powerMonitor.storedCharge_Init
        self.obs[0,0] = np.linalg.norm(self.scObject.hub.sigma_BNInit)
        self.obs[1,0] = np.linalg.norm(self.scObject.hub.omega_BN_BInit)
        self.obs[2,0] = np.linalg.norm(initWheelSpeeds)
        self.obs[3,0] = self.powerMonitor.storedCharge_Init/3600.0



        #   Add all the models to the dynamics task
        self.AddModelToTask(self.dynTaskName, self.scObject)
        self.AddModelToTask(self.spiceTaskName, self.gravFactory.spiceObject)
        self.AddModelToTask(self.dynTaskName, self.densityModel)
        self.AddModelToTask(self.dynTaskName, self.dragEffector)
        self.AddModelToTask(self.dynTaskName, self.simpleNavObject)
        self.AddModelToTask(self.dynTaskName, self.rwStateEffector)
        self.AddModelToTask(self.dynTaskName, self.thrusterSet)
        self.AddModelToTask(self.envTaskName, self.eclipseObject)
        self.AddModelToTask(self.envTaskName, self.solarPanel)
        self.AddModelToTask(self.envTaskName, self.powerMonitor)
        self.AddModelToTask(self.envTaskName, self.powerSink)

        return


    def set_fsw(self):
        '''
        Sets up the attitude guidance stack for the simulation. This simulator runs:
        inertial3Dpoint - Sets the attitude guidance objective to point the main panel at the sun.
        hillPointTask: Sets the attitude guidance objective to point a "camera" angle towards nadir.
        attitudeTrackingError: Computes the difference between estimated and guidance attitudes
        mrpFeedbackControl: Computes an appropriate control torque given an attitude error
        :return:
        '''

        self.processName = self.DynamicsProcessName
        self.processTasksTimeStep = mc.sec2nano(self.fswRate)  # 0.5
        self.dynProc.addTask(self.CreateNewTask("sunPointTask", self.processTasksTimeStep),taskPriority=100)
        self.dynProc.addTask(self.CreateNewTask("nadirPointTask", self.processTasksTimeStep),taskPriority=100)
        self.dynProc.addTask(self.CreateNewTask("mrpControlTask", self.processTasksTimeStep), taskPriority=50)
        self.dynProc.addTask(self.CreateNewTask("rwDesatTask", self.processTasksTimeStep), taskPriority=100)

        #   Specify the vehicle configuration message to tell things what the vehicle inertia is
        vehicleConfigOut = fswMessages.VehicleConfigFswMsg()
        # use the same inertia in the FSW algorithm as in the simulation
        #   Set inertia properties to those of a solid 6U cubeoid:
        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")
        I = [1. / 12. * self.mass * (width ** 2. + depth ** 2.), 0., 0.,
             0., 1. / 12. * self.mass * (depth ** 2. + height ** 2.), 0.,
             0., 0., 1. / 12. * self.mass * (width ** 2. + height ** 2.)]

        vehicleConfigOut.ISCPntB_B = I
        unitTestSupport.setMessage(self.TotalSim,
                                   self.DynamicsProcessName,
                                   "adcs_config_data",
                                   vehicleConfigOut)


        #   Sun pointing configuration
        self.sunPointData = inertial3D.inertial3DConfig()
        self.sunPointWrap = self.setModelDataWrap(self.sunPointData)
        self.sunPointWrap.ModelTag = "sunPoint"
        self.sunPointData.outputDataName = "att_reference"
        self.sunPointData.sigma_R0N = self.initial_conditions.get("sigma_R0N")

        #   Earth pointing configuration
        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = self.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"
        self.hillPointData.outputDataName = "att_reference"
        self.hillPointData.inputNavDataName = self.simpleNavObject.outputTransName
        self.hillPointData.inputCelMessName = self.gravFactory.gravBodies['earth'].bodyInMsgName

        #   Attitude error configuration
        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = self.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"
        self.trackingErrorData.inputNavName = self.simpleNavObject.outputAttName
        # Note: SimBase.DynModels.simpleNavObject.outputAttName = "simple_att_nav_output"
        self.trackingErrorData.inputRefName = "att_reference"
        self.trackingErrorData.outputDataName = "att_guidance"

        # add module that maps the Lr control torque into the RW motor torques
        self.rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
        self.rwMotorTorqueWrap = self.setModelDataWrap(self.rwMotorTorqueConfig)
        self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"
        self.rwMotorTorqueConfig.outputDataName = self.rwStateEffector.InputCmds
        self.rwMotorTorqueConfig.rwParamsInMsgName = "rwConfig"
        self.rwMotorTorqueConfig.inputVehControlName = "commandedControlTorque"
        self.rwMotorTorqueConfig.controlAxes_B = self.initial_conditions.get("controlAxes_B")

        #   Attitude controller configuration
        self.mrpFeedbackControlData = MRP_Feedback.MRP_FeedbackConfig()
        self.mrpFeedbackControlWrap = self.setModelDataWrap(self.mrpFeedbackControlData)
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"
        self.mrpFeedbackControlData.inputGuidName = "att_guidance"
        self.mrpFeedbackControlData.vehConfigInMsgName = "adcs_config_data"
        self.mrpFeedbackControlData.outputDataName = self.rwMotorTorqueConfig.inputVehControlName
        self.mrpFeedbackControlData.K = self.initial_conditions.get("K")
        self.mrpFeedbackControlData.Ki = self.initial_conditions.get("Ki")
        self.mrpFeedbackControlData.P = self.initial_conditions.get("P")
        self.mrpFeedbackControlData.integralLimit = 2. / self.mrpFeedbackControlData.Ki * 0.1

        #   Momentum dumping configuration
        self.thrDesatControlConfig = thrMomentumManagement.thrMomentumManagementConfig()
        self.thrDesatControlWrap = self.setModelDataWrap(self.thrDesatControlConfig)
        self.thrDesatControlWrap.ModelTag = "thrMomentumManagement"
        self.thrDesatControlConfig.hs_min = self.initial_conditions.get("hs_min")  # Nms
        self.thrDesatControlConfig.rwSpeedsInMsgName = self.rwStateEffector.OutputDataString
        self.thrDesatControlConfig.rwConfigDataInMsgName = self.rwConfigMsgName
        self.thrDesatControlConfig.deltaHOutMsgName = "wheelDeltaH"

        # setup the thruster force mapping module
        self.thrForceMappingConfig = thrForceMapping.thrForceMappingConfig()
        self.thrForceMappingWrap = self.setModelDataWrap(self.thrForceMappingConfig)
        self.thrForceMappingWrap.ModelTag = "thrForceMapping"
        self.thrForceMappingConfig.inputVehControlName = self.thrDesatControlConfig.deltaHOutMsgName
        self.thrForceMappingConfig.inputThrusterConfName = self.thrusterConfigMsgName
        self.thrForceMappingConfig.inputVehicleConfigDataName = self.mrpFeedbackControlData.vehConfigInMsgName
        self.thrForceMappingConfig.outputDataName = "delta_p_achievable"
        self.thrForceMappingConfig.controlAxes_B = self.initial_conditions.get("controlAxes_B")
        self.thrForceMappingConfig.thrForceSign = self.initial_conditions.get("thrForceSign")

        self.thrDumpConfig = thrMomentumDumping.thrMomentumDumpingConfig()
        self.thrDumpWrap = self.setModelDataWrap(self.thrDumpConfig)
        self.thrDumpConfig.deltaHInMsgName = self.thrDesatControlConfig.deltaHOutMsgName
        self.thrDumpConfig.thrusterImpulseInMsgName = "delta_p_achievable"
        self.thrDumpConfig.thrusterOnTimeOutMsgName = self.thrusterSet.InputCmds
        self.thrDumpConfig.thrusterConfInMsgName = self.thrusterConfigMsgName
        self.thrDumpConfig.maxCounterValue = self.initial_conditions.get("maxCounterValue")
        self.thrDumpConfig.thrMinFireTime = self.initial_conditions.get("thrMinFireTime")

        #   Add models to tasks
        self.AddModelToTask("sunPointTask", self.sunPointWrap, self.sunPointData)
        self.AddModelToTask("nadirPointTask", self.hillPointWrap, self.hillPointData)

        self.AddModelToTask("mrpControlTask", self.mrpFeedbackControlWrap, self.mrpFeedbackControlData)
        self.AddModelToTask("mrpControlTask", self.trackingErrorWrap, self.trackingErrorData)
        self.AddModelToTask("mrpControlTask", self.rwMotorTorqueWrap, self.rwMotorTorqueConfig)

        self.AddModelToTask("rwDesatTask", self.thrDesatControlWrap, self.thrDesatControlConfig)
        self.AddModelToTask("rwDesatTask", self.thrForceMappingWrap, self.thrForceMappingConfig)
        self.AddModelToTask("rwDesatTask", self.thrDumpWrap, self.thrDumpConfig)


    def set_logging(self):
        '''
        Logs simulation outputs to return as observations. This simulator observes:
        mrp_bn - inertial to body MRP
        error_mrp - Attitude error given current guidance objective
        power_level - current W-Hr from the battery
        r_bn - inertial position of the s/c relative to Earth
        :return:
        '''

        ##  Set the sampling time to the duration of a timestep:
        samplingTime = mc.sec2nano(self.step_duration)

        #   Log planet, sun positions

        #self.TotalSim.logThisMessage("earth_planet_data", samplingTime)
        # self.TotalSim.logThisMessage("sun_planet_data", samplingTime)
        #   Log inertial attitude, position
        self.TotalSim.logThisMessage(self.scObject.scStateOutMsgName, samplingTime)
        self.TotalSim.logThisMessage(self.simpleNavObject.outputTransName,
                                               samplingTime)
        self.TotalSim.logThisMessage(self.simpleNavObject.outputAttName,
                                     samplingTime)
        self.TotalSim.logThisMessage(self.rwStateEffector.OutputDataString,
                                     samplingTime)
        #   Log FSW error portrait
        self.TotalSim.logThisMessage("att_reference", samplingTime)
        self.TotalSim.logThisMessage(self.trackingErrorData.outputDataName, samplingTime)
        # self.TotalSim.logThisMessage(self.mrpFeedbackControlData.outputDataName, samplingTime)

        #   Log system power status
        self.TotalSim.logThisMessage(self.powerMonitor.batPowerOutMsgName,
                                               samplingTime)

        #   Eclipse indicator
        self.TotalSim.logThisMessage(self.solarPanel.sunEclipseInMsgName, samplingTime)

        #   Desat debug parameters
        #self.TotalSim.logThisMessage(self.thrDesatControlConfig.deltaHOutMsgName, samplingTime)

        return

    def run_sim(self, action):
        '''
        Executes the sim for a specified duration given a mode command.
        :param action:
        :param duration:
        :return:
        '''

        self.modeRequest = str(action)

        self.sim_over = False

        currentResetTime = mc.sec2nano(self.simTime)
        if self.modeRequest == "0":
            #print('starting nadir pointing...')
            #   Set up a nadir pointing mode
            self.dynProc.enableAllTasks()
            self.hillPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)

            self.disableTask('sunPointTask')
            self.disableTask('rwDesatTask')

            self.enableTask('nadirPointTask')
            self.enableTask('mrpControlTask')

        elif self.modeRequest == "1":
            #print('starting sun pointing...')
            #   Set up a sun pointing mode
            self.dynProc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)

            self.disableTask('nadirPointTask')
            self.disableTask('rwDesatTask')

            self.enableTask('sunPointTask')
            self.enableTask('mrpControlTask')

        elif self.modeRequest == "2":
            #   Set up a desat mode
            self.dynProc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)

            self.thrDesatControlWrap.Reset(currentResetTime)
            self.thrDumpWrap.Reset(currentResetTime)

            self.disableTask('nadirPointTask')
            self.disableTask('sunPointTask')

            self.enableTask('sunPointTask')
            self.enableTask('mrpControlTask')
            self.enableTask('rwDesatTask')

        self.simTime += self.step_duration
        simulationTime = mc.sec2nano(self.simTime)

        #   Execute the sim
        self.ConfigureStopTime(simulationTime)
        self.ExecuteSimulation()

        #   Pull logged message data and return it as an observation
        simDict = self.pullMultiMessageLogData([
            # 'sun_planet_data.PositionVector',
            # self.scObject.scStateOutMsgName + '.sigma_BN',
            self.scObject.scStateOutMsgName + '.r_BN_N',
            # self.scObject.scStateOutMsgName + '.v_BN_N',
            "att_reference.sigma_RN",
            self.simpleNavObject.outputAttName + '.omega_BN_B',
            self.trackingErrorData.outputDataName + '.sigma_BR',
            self.rwStateEffector.OutputDataString + '.wheelSpeeds',
            self.powerMonitor.batPowerOutMsgName + '.storageLevel',
            self.solarPanel.sunEclipseInMsgName + '.shadowFactor'
        ], [
            # list(range(3)), 
            list(range(3)), 
            # list(range(3)), 
            # list(range(3)), 
            list(range(3)), list(range(3)), list(range(3)), list(range(3)),
            list(range(1)), list(range(1))],
            [
                # 'double','double','double', 
                'double',
            'double','double', 'double','double','double', 'double'], 1)

        #   Observations
        attErr = simDict[self.trackingErrorData.outputDataName + '.sigma_BR']
        attRef = simDict["att_reference.sigma_RN"]
        attRate = simDict[self.simpleNavObject.outputAttName + '.omega_BN_B']
        storedCharge = simDict[self.powerMonitor.batPowerOutMsgName + '.storageLevel']
        eclipseIndicator = simDict[self.solarPanel.sunEclipseInMsgName + '.shadowFactor']
        wheelSpeeds = simDict[self.rwStateEffector.OutputDataString+'.wheelSpeeds']

        #   Debug info
        # sunPosition = simDict['sun_planet_data.PositionVector']
        # inertialAtt = simDict[self.scObject.scStateOutMsgName + '.sigma_BN']
        inertialPos = simDict[self.scObject.scStateOutMsgName + '.r_BN_N']
        # inertialVel = simDict[self.scObject.scStateOutMsgName + '.v_BN_N']

        # debug = np.hstack([inertialAtt[-1,1:4],inertialPos[-1,1:4],inertialVel[-1,1:4],attRef[-1,1:4], sunPosition[-1,1:4]])
        obs = np.hstack([np.linalg.norm(attErr[-1,1:4]), np.linalg.norm(attRate[-1,1:4]), np.linalg.norm(wheelSpeeds[-1,1:4]),
                         storedCharge[-1,1]/3600., eclipseIndicator[-1,1]])
        self.obs = obs.reshape(len(obs), 1)
        self.sim_states = []#debug.reshape(len(debug), 1)

        if np.linalg.norm(inertialPos[-1,1:4]) < (orbitalMotion.REQ_EARTH/1000.):
            self.sim_over = True

        return self.obs, self.sim_states, self.sim_over

    def close_gracefully(self):
        """
        makes sure spice gets shut down right when we close.
        :return:
        """
        self.gravFactory.unloadSpiceKernels()
        return

def create_leoPowerAttSimulator():
    return LEOPowerAttitudeSimulator(0.1, 0.1, 60.)

if __name__=="__main__":
    """
    Test execution of the simulator with random actions and plot the observation space.
    """
    sim = LEOPowerAttitudeSimulator(0.1,1.0, 60.)
    obs = []
    states = []
    normWheelSpeed = []
    actList = []
    from matplotlib import pyplot as plt
    from random import randrange
    plt.figure()

    tFinal = 2*180
    for ind in range(0,tFinal):
        act = 0#randrange(3)
        actList.append(act)
        ob, state, _ = sim.run_sim(act)
        #normWheelSpeed.append(np.linalg.norm(abs(ob[3:6])))
        obs.append(ob)
        states.append(state)
    obs = np.asarray(obs)
    states = np.asarray(states)

    plt.plot(range(0,tFinal),obs[:,0], label="sigma_BR")
    plt.plot(range(0,tFinal),obs[:,1], label="omega_BN")
    plt.plot(range(0,tFinal),obs[:,2], label="omega_rw")
    plt.plot(range(0,tFinal), obs[:,3], label= "J_bat")
    plt.plot(range(0,tFinal),obs[:,4], label="eclipse_ind")
    plt.legend()


    # plt.figure()
    #plt.plot(states[:, 3]/1000., states[:, 4]/1000., label="Orbit")
    #plt.plot(states[:,12]/1000., states[:,13]/1000, label="Sun Position")
    # plt.legend()

    plt.show()





