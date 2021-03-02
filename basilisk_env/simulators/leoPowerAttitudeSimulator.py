from time import sleep
import random
# 3rd party modules
import numpy as np

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion

from Basilisk.simulation import (spacecraft, extForceTorque, simpleNav,
                                 eclipse, exponentialAtmosphere, facetDragDynamicEffector)
from Basilisk.simulation import ephemerisConverter
from Basilisk.utilities import simIncludeGravBody

from Basilisk.simulation import simpleBattery, simplePowerSink, simpleSolarPanel
from Basilisk import __path__
bskPath = __path__[0]

from Basilisk.fswAlgorithms import inertial3D, hillPoint
from Basilisk.fswAlgorithms import attTrackingError
from Basilisk.fswAlgorithms import mrpFeedback, rwMotorTorque


from Basilisk.fswAlgorithms import thrMomentumManagement, thrMomentumDumping, thrForceMapping

from basilisk_env.simulators.dynamics.effectorPrimatives import actuatorPrimatives as ap
from basilisk_env.simulators.initial_conditions import leo_initial_conditions

from Basilisk.architecture import messaging
import Basilisk.architecture.cMsgCInterfacePy as cMsgPy

class LEOPowerAttitudeSimulator(SimulationBaseClass.SimBaseClass):
    """
    Simulates a spacecraft in LEO with atmospheric drag and J2.

    Dynamics Components
    - Forces: J2, Atmospheric Drag w/ COM offset
    - Environment: Exponential density model; eclipse
    - Actuators: ExternalForceTorque
    - Sensors: SimpleNav
    - Systems: SimpleBattery, SimpleSink, SimpleSolarPanel

    FSW Components:
    - mrpFeedback controller
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
    """
    def __init__(self, dynRate, fswRate, step_duration, initial_conditions=None):
        '''
        Creates the simulation, but does not initialize the initial conditions.
        '''
        self.dynRate = dynRate
        self.fswRate = fswRate
        self.step_duration = step_duration

        SimulationBaseClass.SimBaseClass.__init__(self)

        # define class variables that are assigned later on
        self.attRefMsg = None
        self.scRec = None
        self.snTransRec = None
        self.snAttRec = None
        self.rwSpeedRec = None
        self.attRefRec = None
        self.attErrRec = None
        self.powRec = None
        self.powRec = None

        self.simTime = 0.0
        self.sim_over = None

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
        self.setupGatewayMsgs()
        self.set_fsw()

        self.previousPointingGoal = "sunPointTask"

        self.modeRequest = None
        self.InitializeSimulation()

        return

    def set_ICs(self):
        # Sample orbital parameters

        initial_conditions = leo_initial_conditions.sampled_400km_leo_smallsat_tumble()
        
        return initial_conditions

    def set_dynamics(self):
        """
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
        """
        sc_number=0

        #   Spacecraft, Planet Setup
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = 'spacecraft'

        # clear prior gravitational body and SPICE setup definitions
        sleep(random.randint(0,3))
        self.gravFactory = simIncludeGravBody.gravBodyFactory()

        self.gravFactory.createSun()
        self.sun = 0
        planet = self.gravFactory.createEarth()
        self.earth = 1
        planet.isCentralBody = True  # ensure this is the central gravitational body
        mu = planet.mu
        # attach gravity model to spaceCraftPlus
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(list(self.gravFactory.gravBodies.values()))

        # setup Spice interface for some solar system bodies
        timeInitString = '2021 MAY 04 07:47:48.965 (UTC)'
        self.gravFactory.createSpiceInterface(bskPath + '/supportData/EphemerisData/'
                                              , timeInitString
                                              )

        self.gravFactory.spiceObject.zeroBase = "earth"  # Make sure that the Earth is the zero base

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
        self.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsg)
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

        self.dragEffector.atmoDensInMsg.subscribeTo(self.densityModel.envOutMsgs[-1])
        self.scObject.addDynamicEffector(self.dragEffector)

        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.eclipseObject.addPlanetToModel(self.gravFactory.spiceObject.planetStateOutMsgs[self.earth])
        self.eclipseObject.sunInMsg.subscribeTo(self.gravFactory.spiceObject.planetStateOutMsgs[self.sun])

        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(self.gravFactory.spiceObject.planetStateOutMsgs[self.earth])

        #   Disturbance Torque Setup
        disturbance_magnitude = self.initial_conditions.get("disturbance_magnitude")
        disturbance_vector = self.initial_conditions.get("disturbance_vector")
        unit_disturbance = disturbance_vector/np.linalg.norm(disturbance_vector)
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.extTorquePntB_B = disturbance_magnitude * disturbance_vector
        self.extForceTorqueObject.ModelTag = 'DisturbanceTorque'
        # self.extForceTorqueObject.cmdForceInertialInMsg = 'disturbance_torque'
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

        # Add reaction wheels to the spacecraft
        self.rwStateEffector, rwFactory, initWheelSpeeds = ap.balancedHR16Triad(useRandom=True,randomBounds=(-800,800))
        # Change the wheel speeds
        rwFactory.rwList["RW1"].Omega = self.initial_conditions.get("wheelSpeeds")[0]*mc.RPM #rad/s
        rwFactory.rwList["RW2"].Omega = self.initial_conditions.get("wheelSpeeds")[1]*mc.RPM #rad/s
        rwFactory.rwList["RW3"].Omega = self.initial_conditions.get("wheelSpeeds")[2]*mc.RPM #rad/s
        initWheelSpeeds = self.initial_conditions.get("wheelSpeeds")
        rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.rwConfigMsg = rwFactory.getConfigMessage()

        #   Add thrusters to the spacecraft
        self.thrusterSet, thrFactory = ap.idealMonarc1Octet()
        thrModelTag = "ACSThrusterDynamics"
        self.thrusterConfigMsg = thrFactory.getConfigMessage()
        thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        #   Add simpleNav as a mock estimator to the spacecraft
        self.simpleNavObject = simpleNav.SimpleNav()
        self.simpleNavObject.ModelTag = 'SimpleNav'
        self.simpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        #   Power Setup
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = 'solarPanel' + str(sc_number)
        self.solarPanel.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.solarPanel.sunEclipseInMsg.subscribeTo(self.eclipseObject.eclipseOutMsgs[sc_number])
        self.solarPanel.sunInMsg.subscribeTo(self.gravFactory.spiceObject.planetStateOutMsgs[self.sun])
        self.solarPanel.setPanelParameters(unitTestSupport.np2EigenVectorXd(self.initial_conditions.get("nHat_B")), self.initial_conditions.get("panelArea"), self.initial_conditions.get("panelEfficiency"))

        self.powerSink = simplePowerSink.SimplePowerSink()
        self.powerSink.ModelTag = "powerSink" + str(sc_number)
        self.powerSink.nodePowerOut = self.powerDraw  # Watts

        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.storageCapacity = self.initial_conditions.get("storageCapacity")
        self.powerMonitor.storedCharge_Init = self.initial_conditions.get("storedCharge_Init")
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.powerSink.nodePowerOutMsg)

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
        """
        Sets up the attitude guidance stack for the simulation. This simulator runs:
        inertial3Dpoint - Sets the attitude guidance objective to point the main panel at the sun.
        hillPointTask: Sets the attitude guidance objective to point a "camera" angle towards nadir.
        attitudeTrackingError: Computes the difference between estimated and guidance attitudes
        mrpFeedbackControl: Computes an appropriate control torque given an attitude error
        rwDesatTask: desaturates the RW devices
        :return:
        """

        self.processName = self.DynamicsProcessName
        self.processTasksTimeStep = mc.sec2nano(self.fswRate)  # 0.5
        self.dynProc.addTask(self.CreateNewTask("sunPointTask", self.processTasksTimeStep), taskPriority=100)
        self.dynProc.addTask(self.CreateNewTask("nadirPointTask", self.processTasksTimeStep), taskPriority=100)
        self.dynProc.addTask(self.CreateNewTask("mrpControlTask", self.processTasksTimeStep), taskPriority=50)
        self.dynProc.addTask(self.CreateNewTask("rwDesatTask", self.processTasksTimeStep), taskPriority=100)

        #   Specify the vehicle configuration message to tell things what the vehicle inertia is
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        # use the same inertia in the FSW algorithm as in the simulation
        #   Set inertia properties to those of a solid 6U cubeoid:
        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")
        I = [1. / 12. * self.mass * (width ** 2. + depth ** 2.), 0., 0.,
             0., 1. / 12. * self.mass * (depth ** 2. + height ** 2.), 0.,
             0., 0., 1. / 12. * self.mass * (width ** 2. + height ** 2.)]
        vehicleConfigOut.ISCPntB_B = I
        # adcs_config_data -> vcConfigMsg
        self.vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

        #   Sun pointing configuration
        self.sunPointData = inertial3D.inertial3DConfig()
        self.sunPointWrap = self.setModelDataWrap(self.sunPointData)
        self.sunPointWrap.ModelTag = "sunPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.sunPointData.attRefOutMsg, self.attRefMsg)
        self.sunPointData.sigma_R0N = self.initial_conditions.get("sigma_R0N")

        #   Earth pointing configuration
        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = self.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.hillPointData.attRefOutMsg, self.attRefMsg)
        self.hillPointData.transNavInMsg.subscribeTo(self.simpleNavObject.transOutMsg)
        self.hillPointData.celBodyInMsg.subscribeTo(self.ephemConverter.ephemOutMsgs[0])

        #   Attitude error configuration
        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = self.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"
        self.trackingErrorData.attNavInMsg.subscribeTo(self.simpleNavObject.attOutMsg)
        self.trackingErrorData.attRefInMsg.subscribeTo(self.attRefMsg)

        #   Attitude controller configuration
        self.mrpFeedbackControlData = mrpFeedback.mrpFeedbackConfig()
        self.mrpFeedbackControlWrap = self.setModelDataWrap(self.mrpFeedbackControlData)
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"
        self.mrpFeedbackControlData.guidInMsg.subscribeTo(self.trackingErrorData.attGuidOutMsg)
        self.mrpFeedbackControlData.vehConfigInMsg.subscribeTo(self.vcConfigMsg)
        self.mrpFeedbackControlData.K = self.initial_conditions.get("K")
        self.mrpFeedbackControlData.Ki = self.initial_conditions.get("Ki")
        self.mrpFeedbackControlData.P = self.initial_conditions.get("P")
        self.mrpFeedbackControlData.integralLimit = 2. / self.mrpFeedbackControlData.Ki * 0.1

        # add module that maps the Lr control torque into the RW motor torques
        self.rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
        self.rwMotorTorqueWrap = self.setModelDataWrap(self.rwMotorTorqueConfig)
        self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.rwMotorTorqueConfig.rwMotorTorqueOutMsg)
        self.rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(self.rwConfigMsg)
        self.rwMotorTorqueConfig.vehControlInMsg.subscribeTo(self.mrpFeedbackControlData.cmdTorqueOutMsg)
        self.rwMotorTorqueConfig.controlAxes_B = self.initial_conditions.get("controlAxes_B")
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.rwMotorTorqueConfig.rwMotorTorqueOutMsg)

        #   Momentum dumping configuration
        self.thrDesatControlConfig = thrMomentumManagement.thrMomentumManagementConfig()
        self.thrDesatControlWrap = self.setModelDataWrap(self.thrDesatControlConfig)
        self.thrDesatControlWrap.ModelTag = "thrMomentumManagement"
        self.thrDesatControlConfig.hs_min = self.initial_conditions.get("hs_min")  # Nms
        self.thrDesatControlConfig.rwSpeedsInMsg.subscribeTo(self.rwStateEffector.rwSpeedOutMsg)
        self.thrDesatControlConfig.rwConfigDataInMsg.subscribeTo(self.rwConfigMsg)

        # setup the thruster force mapping module
        self.thrForceMappingConfig = thrForceMapping.thrForceMappingConfig()
        self.thrForceMappingWrap = self.setModelDataWrap(self.thrForceMappingConfig)
        self.thrForceMappingWrap.ModelTag = "thrForceMapping"
        self.thrForceMappingConfig.cmdTorqueInMsg.subscribeTo(self.thrDesatControlConfig.deltaHOutMsg)
        self.thrForceMappingConfig.thrConfigInMsg.subscribeTo(self.thrusterConfigMsg)
        self.thrForceMappingConfig.vehConfigInMsg.subscribeTo(self.vcConfigMsg)
        self.thrForceMappingConfig.controlAxes_B = self.initial_conditions.get("controlAxes_B")
        self.thrForceMappingConfig.thrForceSign = self.initial_conditions.get("thrForceSign")

        self.thrDumpConfig = thrMomentumDumping.thrMomentumDumpingConfig()
        self.thrDumpWrap = self.setModelDataWrap(self.thrDumpConfig)
        self.thrDumpConfig.deltaHInMsg.subscribeTo(self.thrDesatControlConfig.deltaHOutMsg)
        self.thrDumpConfig.thrusterImpulseInMsg.subscribeTo(self.thrForceMappingConfig.thrForceCmdOutMsg)
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrDumpConfig.thrusterOnTimeOutMsg)
        self.thrDumpConfig.thrusterConfInMsg.subscribeTo(self.thrusterConfigMsg)
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

    def setupGatewayMsgs(self):
        """create C-wrapped gateway messages such that different modules can write to this message
        and provide a common input msg for down-stream modules"""
        self.attRefMsg = cMsgPy.AttRefMsg_C()

        self.zeroGateWayMsgs()

    def zeroGateWayMsgs(self):
        """Zero all the FSW gateway message payloads"""
        self.attRefMsg.write(messaging.AttRefMsgPayload())

    def run_sim(self, action):
        """
        Executes the sim for a specified duration given a mode command.
        :param action:
        :param duration:
        :return:
        """

        self.modeRequest = str(action)

        self.sim_over = False

        currentResetTime = mc.sec2nano(self.simTime)
        self.zeroGateWayMsgs()
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

        #   Observations
        attErr = self.trackingErrorData.attGuidOutMsg.read().sigma_BR
        attRate = self.simpleNavObject.attOutMsg.read().omega_BN_B
        storedCharge = self.powerMonitor.batPowerOutMsg.read().storageLevel
        eclipseIndicator = self.eclipseObject.eclipseOutMsgs[0].read().shadowFactor
        wheelSpeeds = self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds

        #   Debug info
        # sunPosition = self.gravFactory.spiceObject.planetStateOutMsgs[self.sun].read().PositionVector
        # inertialAtt = self.gravFactory.spiceObject.planetStateOutMsgs[self.earth].read().PositionVector
        inertialPos = self.scObject.scStateOutMsg.read().r_BN_N
        # inertialVel = self.scObject.scStateOutMsg.read().v_BN_N

        obs = np.hstack([np.linalg.norm(attErr), np.linalg.norm(attRate), np.linalg.norm(wheelSpeeds),
                         storedCharge/3600., eclipseIndicator])
        self.obs = obs.reshape(len(obs), 1)
        self.sim_states = []#debug.reshape(len(debug), 1)

        if np.linalg.norm(inertialPos) < (orbitalMotion.REQ_EARTH/1000.):
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





