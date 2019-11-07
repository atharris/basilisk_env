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

from Basilisk.fswAlgorithms import inertial3D, hillPoint
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import MRP_Feedback, rwMotorTorque
from Basilisk.fswAlgorithms import fswMessages
from Basilisk.fswAlgorithms import inertialUKF
from Basilisk.fswAlgorithms import thrMomentumManagement, thrMomentumDumping, thrForceMapping, thrFiringSchmitt

from basilisk_env.simulators.dynamics.effectorPrimatives import actuatorPrimatives as ap

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
    def __init__(self, dynRate, fswRate, step_duration):
        '''
        Creates the simulation, but does not initialize the initial conditions.
        '''
        self.dynRate = dynRate
        self.fswRate = fswRate
        self.step_duration = step_duration

        SimulationBaseClass.SimBaseClass.__init__(self)
        self.TotalSim.terminateSimulation()

        self.simTime = 0.0

        self.obs = np.zeros([5,])

        self.DynModels = []
        self.FSWModels = []

        #   Initialize the dynamics+fsw task groups, modules

        self.DynamicsProcessName = 'DynamicsProcess' #Create simulation process name
        self.dynProc = self.CreateNewProcess(self.DynamicsProcessName) #Create process
        self.dynTaskName = 'DynTask'
        self.dynTask = self.dynProc.addTask(self.CreateNewTask(self.dynTaskName, mc.sec2nano(self.dynRate)))

        self.FSWProcessName = "FSWProcess" #Create simulation process name
        self.fswProc = self.CreateNewProcess(self.FSWProcessName) #Create processes
        self.fswTaskName = 'FswTask'
        self.fswTask = self.fswProc.addTask(self.CreateNewTask(self.fswTaskName, mc.sec2nano(self.fswRate)))

        self.obs = np.zeros([5,1])
        self.sim_states = np.zeros([11,1])

        self.set_dynamics()
        self.set_fsw()

        self.set_logging()

        self.modeRequest = None
        #   Initialize interfaces for fsw, dynamics
        self.dyn2FSWInterface = sim_model.SysInterface()
        self.fsw2DynInterface = sim_model.SysInterface()

        # Discover interfaces between processes
        self.dyn2FSWInterface.addNewInterface(self.DynamicsProcessName, self.FSWProcessName)
        self.fsw2DynInterface.addNewInterface(self.FSWProcessName, self.DynamicsProcessName)
        self.dynProc.addInterfaceRef(self.dyn2FSWInterface)
        self.fswProc.addInterfaceRef(self.fsw2DynInterface)

        self.InitializeSimulationAndDiscover()

        return


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

        planet = self.gravFactory.createEarth()
        planet.isCentralBody = True  # ensure this is the central gravitational body
        mu = planet.mu
        # attach gravity model to spaceCraftPlus
        self.scObject.gravField.gravBodies = spacecraftPlus.GravBodyVector(list(self.gravFactory.gravBodies.values()))

        #   Stick the sun 1AU away
        self.sunMessage = simMessages.SpicePlanetStateSimMsg()
        self.sunMessage.PositionVector = [orbitalMotion.AU, 0,0]
        self.sunMessage.VelocityVector = [0,0,0]

        unitTestSupport.setMessage(self.TotalSim,
                                   self.DynamicsProcessName,
                                   "sun_planet_data",
                                   self.sunMessage)

        #   setup orbit using orbitalMotion library
        oe = orbitalMotion.ClassicElements()
        oe.a = 6371 * 1000.0 + 300. * 1000
        oe.e = 0.0
        oe.i = 0.0 * mc.D2R

        oe.Omega = 0.0 * mc.D2R
        oe.omega = 0.0 * mc.D2R
        oe.f = 0.0 * mc.D2R
        rN, vN = orbitalMotion.elem2rv(mu, oe)

        n = np.sqrt(mu / oe.a / oe.a / oe.a)
        P = 2. * np.pi / n

        I = [900., 0., 0.,
             0., 800., 0.,
             0., 0., 600.]

        self.scObject.hub.mHub = 6.0 # kg - spacecraft mass
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        self.scObject.hub.sigma_BNInit = [[0.1], [0.1], [-0.1]]  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.01]]

        self.sim_states[0:3,0] = np.asarray(self.scObject.hub.sigma_BNInit).flatten()
        self.sim_states[3:6,0] = np.asarray(self.scObject.hub.r_CN_NInit).flatten()
        self.sim_states[6:9,0] = np.asarray(self.scObject.hub.v_CN_NInit).flatten()

        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsgName)
        self.densityModel.baseDensity = 1.220#  kg/m^3
        self.densityModel.scaleHeight = 8e3 #   m

        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        #   Set up the goemetry of a 6U cubesat
        self.dragEffector.addFacet(0.2*0.3, 2.2, [1,0,0], [0.05, 0.0, 0])
        self.dragEffector.addFacet(0.2*0.3, 2.2, [-1,0,0], [0.05, 0.0, 0])
        self.dragEffector.addFacet(0.1*0.2, 2.2, [0,1,0], [0, 0.15, 0])
        self.dragEffector.addFacet(0.1*0.2, 2.2, [0,-1,0], [0, -0.15, 0])
        self.dragEffector.addFacet(0.1*0.3, 2.2, [0,0,1], [0,0, 0.1])
        self.dragEffector.addFacet(0.1*0.3, 2.2, [0,0,-1], [0, 0, -0.1])
        self.dragEffector.atmoDensInMsgName = self.densityModel.envOutMsgNames[-1]

        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPositionMsgName(self.scObject.scStateOutMsgName)
        self.eclipseObject.addPlanetName('earth')

        #   Actuator/Sensor Setup
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.ModelTag = 'ControlTorque'
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

        # Add reaction wheels to the spacecraft
        self.rwStateEffector, rwFactory = ap.balancedHR16Triad()
        self.rwStateEffector.InputCmds = "rwTorqueCommand"
        rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.rwConfigMsgName = "rwConfig"
        unitTestSupport.setMessage(self.TotalSim, self.FSWProcessName, self.rwConfigMsgName, rwFactory.getConfigMessage(), msgStrName="RWArrayConfigFswMsg")

        #   Add thrusters to the spacecraft
        self.thrusterSet, thrFactory = ap.idealMonarc1Octet()
        self.thrusterSet.InputCmds = "rwDesatTimeOnCmd"
        thrModelTag = "ACSThrusterDynamics"
        thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)
        self.thrusterConfigMsgName = "thrusterConfig"
        unitTestSupport.setMessage(self.TotalSim, self.FSWProcessName, self.thrusterConfigMsgName, rwFactory.getConfigMessage(), msgStrName="THRArrayConfigFswMsg")

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
        self.solarPanel.setPanelParameters(unitTestSupport.np2EigenVectorXd(np.array([1,0,0])), 0.2*0.3, 0.20)
        self.solarPanel.nodePowerOutMsgName = "panelPowerMsg" + str(sc_number)

        self.powerSink = simplePowerSink.SimplePowerSink()
        self.powerSink = simplePowerSink.SimplePowerSink()
        self.powerSink.ModelTag = "powerSink" + str(sc_number)
        self.powerSink.nodePowerOut = -3.  # Watts
        self.powerSink.nodePowerOutMsgName = "powerSinkMsg" + str(sc_number)

        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.batPowerOutMsgName = "powerMonitorMsg"
        self.powerMonitor.storageCapacity = 10.0 * 3600.
        self.powerMonitor.storedCharge_Init = 10.0 * 3600.
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsgName)
        self.powerMonitor.addPowerNodeToModel(self.powerSink.nodePowerOutMsgName)

        self.sim_states[9,0] = self.powerMonitor.storedCharge_Init
        self.obs[3,0] = self.powerMonitor.storedCharge_Init

        #   Add all the models to the dynamics task
        self.AddModelToTask(self.dynTaskName, self.scObject)
        #self.AddModelToTask(self.dynTaskName, self.gravFactory.spiceObject)
        self.AddModelToTask(self.dynTaskName, self.densityModel)
        self.AddModelToTask(self.dynTaskName, self.dragEffector)
        self.AddModelToTask(self.dynTaskName, self.simpleNavObject)
        self.AddModelToTask(self.dynTaskName, self.extForceTorqueObject)
        self.AddModelToTask(self.dynTaskName, self.rwStateEffector)
        self.AddModelToTask(self.dynTaskName, self.thrusterSet)
        self.AddModelToTask(self.dynTaskName, self.eclipseObject)
        self.AddModelToTask(self.dynTaskName, self.solarPanel)
        self.AddModelToTask(self.dynTaskName, self.powerMonitor)
        self.AddModelToTask(self.dynTaskName, self.powerSink)

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

        self.processName = self.FSWProcessName
        self.processTasksTimeStep = mc.sec2nano(self.fswRate)  # 0.5
        self.fswProc.addTask(self.CreateNewTask("inertial3DPointTask", self.processTasksTimeStep),10)
        self.fswProc.addTask(self.CreateNewTask("nadirPointTask", self.processTasksTimeStep),10)
        self.fswProc.addTask(self.CreateNewTask("mrpControlTask", self.processTasksTimeStep), 10)
        self.fswProc.addTask(self.CreateNewTask("rwDesatTask", self.processTasksTimeStep), 10)

        #   Specify the vehicle configuration message to tell things what the vehicle inertia is
        vehicleConfigOut = fswMessages.VehicleConfigFswMsg()
        # use the same inertia in the FSW algorithm as in the simulation
        #   Set inertia properties to those of a solid 6U cubeoid:
        I = [900., 0., 0.,
             0., 800., 0.,
             0., 0., 600.]

        vehicleConfigOut.ISCPntB_B = I
        unitTestSupport.setMessage(self.TotalSim,
                                   self.FSWProcessName,
                                   "adcs_config_data",
                                   vehicleConfigOut)

        #
        self.inertial3DData = inertial3D.inertial3DConfig()
        self.inertial3DWrap = self.setModelDataWrap(self.inertial3DData)
        self.inertial3DWrap.ModelTag = "inertial3D"
        self.inertial3DData.sigma_R0N = [1.0, 0., 0.]
        self.inertial3DData.outputDataName = "att_reference"

        #   Hill point attitude guidance configuration
        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = self.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"
        self.hillPointData.outputDataName = "att_reference"
        self.hillPointData.inputNavDataName = self.simpleNavObject.outputTransName
        self.hillPointData.inputCelMessName = self.gravFactory.gravBodies['earth'].bodyInMsgName[:-1]

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
        controlAxes_B = [1, 0, 0,
                         0, 1, 0,
                         0, 0, 1]
        self.rwMotorTorqueConfig.controlAxes_B = controlAxes_B

        #   Attitude controller configuration
        self.mrpFeedbackControlData = MRP_Feedback.MRP_FeedbackConfig()
        self.mrpFeedbackControlWrap = self.setModelDataWrap(self.mrpFeedbackControlData)
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"
        self.mrpFeedbackControlData.inputGuidName = "att_guidance"
        self.mrpFeedbackControlData.vehConfigInMsgName = "adcs_config_data"
        self.mrpFeedbackControlData.outputDataName = self.rwMotorTorqueConfig.inputVehControlName
        self.mrpFeedbackControlData.K = 3.5
        self.mrpFeedbackControlData.Ki = -1.0  # Note: make value negative to turn off integral feedback
        self.mrpFeedbackControlData.P = 30
        self.mrpFeedbackControlData.integralLimit = 2. / self.mrpFeedbackControlData.Ki * 0.1

        #   Momentum dumping configuration
        self.thrDesatControlConfig = thrMomentumManagement.thrMomentumManagementConfig()
        self.thrDesatControlWrap = self.setModelDataWrap(self.thrDesatControlConfig)
        self.thrDesatControlWrap.ModelTag = "thrMomentumManagement"
        self.thrDesatControlConfig.hs_min = 100. / 6000. * 100.  # Nms
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
        self.thrForceMappingConfig.controlAxes_B = controlAxes_B
        self.thrForceMappingConfig.thrForceSign = 1

        self.thrDumpConfig = thrMomentumDumping.thrMomentumDumpingConfig()
        self.thrDumpWrap = self.setModelDataWrap(self.thrDesatControlConfig)
        self.thrDumpConfig.deltaHInMsgName = self.thrDesatControlConfig.deltaHOutMsgName
        self.thrDumpConfig.thrusterImpulseInMsgName = "delta_p_achievable"
        self.thrDumpConfig.thrusterOnTimeOutMsgName = self.thrusterSet.InputCmds
        self.thrDumpConfig.thrusterConfInMsgName = self.thrusterConfigMsgName

        #   Add models to tasks
        self.AddModelToTask("inertial3DPointTask", self.inertial3DWrap, self.inertial3DData)
        self.AddModelToTask("nadirPointTask", self.hillPointWrap, self.hillPointData)
        self.AddModelToTask("mrpControlTask", self.mrpFeedbackControlWrap, self.mrpFeedbackControlData)
        self.AddModelToTask("mrpControlTask", self.trackingErrorWrap, self.trackingErrorData)
        self.AddModelToTask("mrpControlTask", self.rwMotorTorqueWrap, self.rwMotorTorqueConfig)

        self.AddModelToTask("rwDesatTask", self.thrDesatControlWrap, self.thrDesatControlConfig)
        self.AddModelToTask("rwDesatTask", self.thrForceMappingWrap, self.thrForceMappingConfig)
        self.AddModelToTask("rwDesatTask", self.thrDumpWrap, self.thrDumpConfig)


        #   Hill pointing task; should point nadir.
        self.createNewEvent("0", mc.sec2nano(self.fswRate), True,
                               ["self.modeRequest == 'hillPoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.dynProc.enableAllTasks()",
                                "self.enableTask('hillPointTask')",
                                "self.enableTask('mrpControlTask')"])
        #   Sun pointing task; should point towards the inertial `1` axis.
        self.createNewEvent("1", mc.sec2nano(self.fswRate), True,
                               ["self.modeRequest == 'sunSafePoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.dynProc.enableAllTasks()",
                                "self.enableTask('inertial3DPointTask')",
                                "self.enableTask('mrpControlTask')"])
        #   Reaction wheel desat task; points towards the "1" axis while pulsing thrusters to reduce wheel biases.
        self.createNewEvent("2", mc.sec2nano(self.fswRate), True,
                               ["self.modeRequest == 'desatPoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.dynProc.enableAllTasks()",
                                "self.enableTask('inertial3DPointTask')",
                                "self.enableTask('mrpControlTask')",
                                "self.enableTask('rwDesatTask')"])

    def set_logging(self):
        '''
        Logs simulation outputs to return as observations. This simulator observes:
        mrp_bn - inertial to body MRP
        error_mrp - Attitude error given current guidance objective
        power_level - current W-Hr from the battery
        r_bn - inertial position of the s/c relative to Earth
        :return:
        '''

        simModuleId = 1

        msgDict = {}

        self.stateMsgId = simulationArchTypes.SubscribeToMessage(self.scObject.scStateOutMsgName, 
                                                simMessages.SCPlusStatesSimMsg(),
                                                simModuleId)

        ##  Set the sampling time to the duration of a timestep:
        samplingTime = mc.sec2nano(self.step_duration)
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
        self.TotalSim.logThisMessage(self.mrpFeedbackControlData.outputDataName, samplingTime)

        #   Log system power status
        self.TotalSim.logThisMessage(self.powerMonitor.batPowerOutMsgName,
                                               samplingTime)

        #   Eclipse indicator
        self.TotalSim.logThisMessage(self.solarPanel.sunEclipseInMsgName, samplingTime)

        return

    def run_sim(self, action):
        '''
        Executes the sim for a specified duration given a mode command.
        :param action:
        :param duration:
        :return:
        '''

        self.modeRequest = str(action)
        if action==2:
            #   Need to reset the wheel desaturation system or things will break.
            reset_time = mc.sec2nano(self.simTime)
            #self.thrDesatControlConfig.thrDumpingCounter = 0
            #self.thrDumpWrap.Reset(reset_time)
        self.simTime += self.step_duration
        simulationTime = mc.sec2nano(self.simTime)

        #   Execute the sim
        self.ConfigureStopTime(simulationTime)
        self.ExecuteSimulation()

        #   Pull logged message data and return it as an observation
        simDict = self.pullMultiMessageLogData([
            self.scObject.scStateOutMsgName + '.sigma_BN',
            self.scObject.scStateOutMsgName + '.r_BN_N',
            self.scObject.scStateOutMsgName + '.v_BN_N',
            self.trackingErrorData.outputDataName + '.sigma_BR',
            self.rwStateEffector.OutputDataString + '.wheelSpeeds',
            self.powerMonitor.batPowerOutMsgName + '.storageLevel',
            self.solarPanel.sunEclipseInMsgName + '.shadowFactor'
        ], [list(range(3)),list(range(3)),list(range(3)), list(range(3)), list(range(3)),list(range(1)), list(range(1))],1)

        attErr = simDict[self.trackingErrorData.outputDataName + '.sigma_BR']
        storedCharge = simDict[self.powerMonitor.batPowerOutMsgName + '.storageLevel']
        eclipseIndicator = simDict[self.solarPanel.sunEclipseInMsgName + '.shadowFactor']
        wheelSpeeds = simDict[self.rwStateEffector.OutputDataString+'.wheelSpeeds']

        inertialAtt = simDict[self.scObject.scStateOutMsgName + '.sigma_BN']
        inertialPos = simDict[self.scObject.scStateOutMsgName + '.r_BN_N']
        inertialVel = simDict[self.scObject.scStateOutMsgName + '.v_BN_N']

        debug = np.hstack([inertialAtt[-1,1:4],inertialPos[-1,1:4],inertialVel[-1,1:4]])
        obs = np.hstack([attErr[-1,1:4], wheelSpeeds[-1,1:4], storedCharge[-1,1]/3600., eclipseIndicator[-1,1]])
        self.obs = obs.reshape(len(obs), 1)
        self.sim_states = debug.reshape(len(debug), 1)

        return obs#, self.sim_states


if __name__=="__main__":
    sim = LEOPowerAttitudeSimulator(1.0, 1.0, 60.0)
    obs = []
    from matplotlib import pyplot as plt
    plt.figure()
    for ind in range(0,1*60):
        obs.append(sim.run_sim(2))
    obs = np.asarray(obs)
    plt.plot(range(0,1*60),obs[:,0], range(0,1*60),obs[:,1],range(0,1*60),obs[:,2], label='sigma_BR')
    plt.plot(range(0,1*60), obs[:,3],range(0,1*60),obs[:,4],range(0,1*60),obs[:,5],label="omega_rw")
    plt.plot(range(0,1*60), obs[:,6],label="stored charge")
    plt.plot(range(0,1*60),obs[:,7],label="Eclipse")
    plt.legend()
    plt.show()









