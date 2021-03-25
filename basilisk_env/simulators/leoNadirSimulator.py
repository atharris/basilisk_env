# 3rd party modules
import numpy as np

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import astroFunctions
from Basilisk.simulation import sim_model

from Basilisk.simulation import (spacecraftPlus, gravityEffector, extForceTorque, simple_nav, spice_interface,
                                eclipse, imu_sensor, exponentialAtmosphere, facetDragDynamicEffector, planetEphemeris, reactionWheelStateEffector,
                                 thrusterDynamicEffector, simpleInstrument, simpleStorageUnit, partitionedStorageUnit, spaceToGroundTransmitter, groundLocation)
from Basilisk.simulation import simMessages
from Basilisk.utilities import simIncludeRW, simIncludeGravBody, simIncludeThruster
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import simulationArchTypes
from Basilisk.simulation import simpleBattery, simplePowerSink, simpleSolarPanel, ReactionWheelPower
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

class LEONadirSimulator(SimulationBaseClass.SimBaseClass):
    '''
    Simulates a spacecraft in LEO with atmospheric drag and J2.

    Dynamics Components
    - Forces: J2, Atmospheric Drag w/ COM offset
    - Environment: Exponential density model; eclipse
    - Actuators: ExternalForceTorque
    - Sensors: SimpleNav
    - Power System: SimpleBattery, SimpleSink, SimpleSolarPanel
    - Data Management System: spaceToGroundTransmitter, simpleStorageUnit, simpleInstrument

    FSW Components:
    - MRP_Feedback controller
    - inertial3d (sun pointing), hillPoint (nadir pointing)
    - Desat

    Action Space (discrete, 0 or 1):
    0 - Imaging mode
    1 - Charging mode
    2 - Desat mode
    3 - Downlink mode

    Observation Space:
    Inertial position and velocity - indices 0-5
    Attitude error and attitude rate - indices 6-7
    Reaction wheel speeds - indices 8-11
    Battery charge - indices 12
    Eclipse indicator - indices 13
    Stored data onboard spacecraft - indices 14
    Data transmitted over interval - indices 15
    Amount of time ground stations were accessible (s) - 16-22
    Percent through planning interval - 23

    Reward Function:
    r = +1 for each MB downlinked and no failure
    r = +1 for each MB downlinked and no failure and +1 if t > t_max
    r = - 1000 if failure (battery drained, buffer overflow, reaction wheel speeds over max)

    :return:
    '''

    def __init__(self, dynRate, fswRate, step_duration, initial_conditions=None):
        '''
        Creates the simulation, but does not initialize the initial conditions.
        '''
        self.dynRate = dynRate
        self.fswRate = fswRate

        self.step_duration = 2*step_duration

        SimulationBaseClass.SimBaseClass.__init__(self)
        self.TotalSim.terminateSimulation()

        self.simTime = 0.0

        # If no initial conditions are defined yet, set ICs
        if initial_conditions == None:
            self.initial_conditions=self.set_ICs()
        # If ICs were passed through, use the ones that were passed through
        else:
            self.initial_conditions = initial_conditions

        self.DynModels = []
        self.FSWModels = []

        #   Initialize the dynamics and fsw task groups and modules
        self.DynamicsProcessName = 'DynamicsProcess' # Create simulation process name
        self.dynProc = self.CreateNewProcess(self.DynamicsProcessName) # Create process
        self.dynTaskName = 'DynTask'
        self.spiceTaskName = 'SpiceTask'
        self.envTaskName = 'EnvTask'
        self.dynTask = self.dynProc.addTask(self.CreateNewTask(self.dynTaskName, mc.sec2nano(self.dynRate)))
        self.spiceTask = self.dynProc.addTask(self.CreateNewTask(self.spiceTaskName, mc.sec2nano(self.dynRate)))
        self.envTask = self.dynProc.addTask(self.CreateNewTask(self.envTaskName, mc.sec2nano(self.dynRate)))

        self.obs = np.zeros([23,1])
        self.obs_full = np.zeros([23,1])
        self.curr_step = 0
        self.max_steps = 0
        self.max_length = 0
        # self.sim_states = np.zeros([11,1])

        self.set_dynamics()
        self.set_fsw()

        self.set_logging()
        self.previousPointingGoal = "sunPointTask"

        self.modeRequest = None
        self.InitializeSimulationAndDiscover()

        return

    def __del__(self):
        self.close_gracefully()
        print('Destructor called, simulation deleted')

    def set_ICs(self):
        # Sample orbital parameters
        oe, rN,vN = leo_orbit.sampled_500km_boulder_gs()

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
            # "disturbance_magnitude": 1e-6,
            "disturbance_magnitude": 2e-3,
            "disturbance_vector": np.random.standard_normal(3),

            # Reaction Wheel speeds
            # "wheelSpeeds": uniform(-400,400,3), # RPM
            "wheelSpeeds": uniform(-4000*mc.RPM,4000*mc.RPM,3), # rad/s

            # Solar Panel Parameters
            "nHat_B": np.array([0,1,0]),
            "panelArea": 2*1.0*0.5,
            "panelEfficiency": 0.20,

            # Power Sink Parameters
            "instrumentPowerDraw": -30.0, # W, Assuming 50 W imager - order of Magnitude from Harris Spaceview line
            "transmitterPowerDraw": -15.0, # W
            "rwBasePower": 0.4, # W, Note the opposite convention
            "rwMechToElecEfficiency": 0.0,
            "rwElecToMechEfficiency": 0.5,

            # Battery Parameters
            "batteryStorageCapacity": 80.0 * 3600.,
            "storedCharge_Init": np.random.uniform(30.*3600., 50.*3600., 1)[0],

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
            #"maxCounterValue": 4,
            "maxCounterValue": 8,
            "thrMinFireTime": 0.002, #   Seconds
            # "thrMinFireTime": 0.2, #   Seconds

            # Ground station - Located in Boulder, CO
            "boulderGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "boulderGroundStationLat": np.radians(40.009971), # 40.0150 N Latitude
            "boulderGroundStationLong": np.radians(-105.243895), # 105.2705 W Longitude
            "boulderGroundStationAlt": 1624, # Altitude
            "boulderMinimumElevation": np.radians(10.),
            "boulderMaximumRange": 1e9,

            # Ground station - Located in Merritt Island, FL
            "merrittGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "merrittGroundStationLat": np.radians(28.3181), # 28.3181 N Latitude
            "merrittGroundStationLong": np.radians(-80.6660), # 80.6660 W Longitude
            "merrittGroundStationAlt": 0.9144, # Altitude
            "merrittMinimumElevation": np.radians(10.),
            "merrittMaximumRange": 1e9,

            # Ground station - Located in Singapore, Malaysia
            "singaporeGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "singaporeGroundStationLat": np.radians(1.3521), # 1.3521 N Latitude
            "singaporeGroundStationLong": np.radians(103.8198), # 103.8198 E Longitude
            "singaporeGroundStationAlt": 15, # Altitude, m
            "singaporeMinimumElevation": np.radians(10.),
            "singaporeMaximumRange": 1e9,

            # Ground station - Located in Weilheim, Germany
            "weilheimGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "weilheimGroundStationLat": np.radians(47.8407), # 47.8407 N Latitude
            "weilheimGroundStationLong": np.radians(11.1421), # 11.1421 E Longitude
            "weilheimGroundStationAlt": 563, # Altitude, m
            "weilheimMinimumElevation": np.radians(10.),
            "weilheimMaximumRange": 1e9,

            # Ground station - Located in Santiago, Chile
            "santiagoGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "santiagoGroundStationLat": np.radians(-33.4489), # 33.4489 S Latitude
            "santiagoGroundStationLong": np.radians(-70.6693), # 70.6693 W Longitude
            "santiagoGroundStationAlt": 570, # Altitude, m
            "santiagoMinimumElevation": np.radians(10.),
            "santiagoMaximumRange": 1e9,

            # Ground station - Located in Dongara, Australia
            "dongaraGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "dongaraGroundStationLat": np.radians(-29.2452), # 29.2452 S Latitude
            "dongaraGroundStationLong": np.radians(114.9326), # 114.9326 E Longitude
            "dongaraGroundStationAlt": 34, # Altitude, m
            "dongaraMinimumElevation": np.radians(10.),
            "dongaraMaximumRange": 1e9,

            # Ground station - Located in Hawaii
            "hawaiiGroundStationPlanetRadius": astroFunctions.E_radius*1e3,
            "hawaiiGroundStationLat": np.radians(19.8968), # 19.8968 N Latitude
            "hawaiiGroundStationLong": np.radians(-155.5828), # 155.5828 W Longitude
            "hawaiiGroundStationAlt": 9, # Altitude, m
            "hawaiiMinimumElevation": np.radians(10.),
            "hawaiiMaximumRange": 1e9,

            # Data-generating instrument
            "instrumentBaudRate": 4e6, # baud, 8e6 = 1 MB = 1 image

            # Transmitter
            "transmitterBaudRate": -4e6,   # 4 Mbits/s
            "transmitterPacketSize": -1,   # bits
            "transmitterNumBuffers": 1,

            # Data Storage Unit
            "dataStorageCapacity": 8E9   # bits (8E9 = 1 GB)

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

        # Grab the mass for readability in inertia computation
        mass = self.initial_conditions.get("mass")

        I = [1./12.*mass*(width**2. + depth**2.), 0., 0.,
             0., 1./12.*mass*(depth**2. + height**2.), 0.,
             0., 0.,1./12.*mass*(width**2. + height**2.)]

        self.scObject.hub.mHub = mass # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        sigma_init = self.initial_conditions.get("sigma_init")
        omega_init = self.initial_conditions.get("omega_init")

        self.scObject.hub.sigma_BNInit = sigma_init  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = omega_init

        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsgName)
        self.densityModel.planetRadius = self.initial_conditions.get("planetRadius")
        self.densityModel.baseDensity = self.initial_conditions.get("baseDensity")
        self.densityModel.scaleHeight = self.initial_conditions.get("scaleHeight")

        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        #  Set up the geometry of a small satellite, starting w/ bus
        self.dragEffector.addFacet(width*depth, 2.2, [1,0,0], [height/2, 0.0, 0])
        self.dragEffector.addFacet(width*depth, 2.2, [-1,0,0], [height/2, 0.0, 0])
        self.dragEffector.addFacet(height*width, 2.2, [0,1,0], [0, depth/2, 0])
        self.dragEffector.addFacet(height*width, 2.2, [0,-1,0], [0, -depth/2, 0])
        self.dragEffector.addFacet(height*depth, 2.2, [0,0,1], [0,0, width/2])
        self.dragEffector.addFacet(height*depth, 2.2, [0,0,-1], [0, 0, -width/2])

        # Add solar panels
        self.dragEffector.addFacet(self.initial_conditions.get("panelArea")/2, 2.2, [0,1,0], [0,height,0])
        self.dragEffector.addFacet(self.initial_conditions.get("panelArea")/2, 2.2, [0,-1,0], [0,height,0])

        self.dragEffector.atmoDensInMsgName = self.densityModel.envOutMsgNames[-1]
        self.scObject.addDynamicEffector(self.dragEffector)

        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPositionMsgName(self.scObject.scStateOutMsgName)
        self.eclipseObject.addPlanetName('earth')

        # Disturbance Torque Setup
        disturbance_magnitude = self.initial_conditions.get("disturbance_magnitude")
        disturbance_vector = self.initial_conditions.get("disturbance_vector")
        unit_disturbance = disturbance_vector/np.linalg.norm(disturbance_vector)
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.extTorquePntB_B = disturbance_magnitude * unit_disturbance
        self.extForceTorqueObject.ModelTag = 'DisturbanceTorque'
        self.extForceTorqueObject.cmdForceInertialInMsgName = 'disturbance_torque'
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

        # Add reaction wheels to the spacecraft
        self.rwStateEffector, rwFactory, initWheelSpeeds = ap.balancedHR16Triad(useRandom=False, randomBounds=(-800,800), wheelSpeeds=self.initial_conditions.get("wheelSpeeds"))
        self.rwStateEffector.InputCmds = "rwTorqueCommand"
        rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.rwConfigMsgName = "rwConfig"
        unitTestSupport.setMessage(self.TotalSim, self.DynamicsProcessName, self.rwConfigMsgName, rwFactory.getConfigMessage(), msgStrName="RWArrayConfigFswMsg")

        # Add thrusters to the spacecraft
        self.thrusterSet, thrFactory = ap.idealMonarc1Octet()
        self.thrusterSet.InputCmds = "rwDesatTimeOnCmd"
        thrModelTag = "ACSThrusterDynamics"
        self.thrusterConfigMsgName = "thrusterConfig"
        unitTestSupport.setMessage(self.TotalSim, self.DynamicsProcessName, self.thrusterConfigMsgName, thrFactory.getConfigMessage(), msgStrName="THRArrayConfigFswMsg")
        thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        # Add simpleNav as a mock estimator to the spacecraft
        self.simpleNavObject = simple_nav.SimpleNav()
        self.simpleNavObject.ModelTag = 'SimpleNav'
        self.simpleNavObject.inputStateName = self.scObject.scStateOutMsgName

        # Power setup
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = 'solarPanel' + str(sc_number)
        self.solarPanel.stateInMsgName = self.scObject.scStateOutMsgName
        self.solarPanel.sunEclipseInMsgName = 'eclipse_data_'+str(sc_number)
        self.solarPanel.sunInMsgName = 'sun_planet_data'
        self.solarPanel.setPanelParameters(unitTestSupport.np2EigenVectorXd(self.initial_conditions.get("nHat_B")),
            self.initial_conditions.get("panelArea"), self.initial_conditions.get("panelEfficiency"))
        self.solarPanel.nodePowerOutMsgName = "panelPowerMsg" + str(sc_number)

        # Instrument power sink
        self.instrumentPowerSink = simplePowerSink.SimplePowerSink()
        self.instrumentPowerSink.ModelTag = "insPowerSink" + str(sc_number)
        self.instrumentPowerSink.nodePowerOut = self.initial_conditions.get("instrumentPowerDraw")  # Watts
        self.instrumentPowerSink.nodePowerOutMsgName = "insPowerSinkMsg" + str(sc_number)

        # Transmitter power sink
        self.transmitterPowerSink = simplePowerSink.SimplePowerSink()
        self.transmitterPowerSink.ModelTag = "transPowerSink" + str(sc_number)
        self.transmitterPowerSink.nodePowerOut = self.initial_conditions.get("transmitterPowerDraw")  # Watts
        self.transmitterPowerSink.nodePowerOutMsgName = "transPowerSinkMsg" + str(sc_number)

        # Reaction wheel power sinks
        self.rwPowerList = []
        for c in range(0, 3):
            powerRW = ReactionWheelPower.ReactionWheelPower()
            powerRW.ModelTag = self.scObject.ModelTag
            # powerRW.ModelTag = "rw" + str(c) + "PowerSink"
            powerRW.basePowerNeed = self.initial_conditions.get("rwBasePower")  # baseline power draw, Watts
            powerRW.rwStateInMsgName = self.rwStateEffector.ModelTag + "_rw_config_" + str(c) + "_data"
            powerRW.nodePowerOutMsgName = "rwPower_" + str(c)
            powerRW.mechToElecEfficiency = self.initial_conditions.get("rwMechToElecEfficiency")
            powerRW.elecToMechEfficiency = self.initial_conditions.get("rwElecToMechEfficiency")
            self.AddModelToTask(self.dynTaskName, powerRW, ModelPriority=(987-c))
            self.rwPowerList.append(powerRW)

        # Battery
        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.batPowerOutMsgName = "powerMonitorMsg"
        self.powerMonitor.storageCapacity = self.initial_conditions.get("batteryStorageCapacity")
        self.powerMonitor.storedCharge_Init = self.initial_conditions.get("storedCharge_Init")
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsgName)
        self.powerMonitor.addPowerNodeToModel(self.instrumentPowerSink.nodePowerOutMsgName)
        self.powerMonitor.addPowerNodeToModel(self.transmitterPowerSink.nodePowerOutMsgName)
        for powerRW in self.rwPowerList:
            self.powerMonitor.addPowerNodeToModel(powerRW.nodePowerOutMsgName)

        # Create a Boulder-based ground station
        self.boulderGroundStation = groundLocation.GroundLocation()
        self.boulderGroundStation.ModelTag = "GroundStation1"
        self.boulderGroundStation.currentGroundStateOutMsgName = "GroundStation1_Location"
        self.boulderGroundStation.planetRadius = self.initial_conditions.get("boulderGroundStationPlanetRadius")
        self.boulderGroundStation.specifyLocation(self.initial_conditions.get("boulderGroundStationLat"),
                                                  self.initial_conditions.get("boulderGroundStationLong"),
                                                  self.initial_conditions.get("boulderGroundStationAlt"))
        self.boulderGroundStation.planetInMsgName = planet.bodyInMsgName
        self.boulderGroundStation.minimumElevation = self.initial_conditions.get("boulderMinimumElevation")
        self.boulderGroundStation.maximumRange = self.initial_conditions.get("boulderMaximumRange")
        self.boulderGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create a Merritt-Island ground station (NASA's Near Earth Network)
        self.merrittGroundStation = groundLocation.GroundLocation()
        self.merrittGroundStation.ModelTag = "GroundStation2"
        self.merrittGroundStation.currentGroundStateOutMsgName = "GroundStation2_Location"
        self.merrittGroundStation.planetRadius = self.initial_conditions.get("merrittGroundStationPlanetRadius")
        self.merrittGroundStation.specifyLocation(self.initial_conditions.get("merrittGroundStationLat"),
                                                  self.initial_conditions.get("merrittGroundStationLong"),
                                                  self.initial_conditions.get("merrittGroundStationAlt"))
        self.merrittGroundStation.planetInMsgName = planet.bodyInMsgName
        self.merrittGroundStation.minimumElevation = self.initial_conditions.get("merrittMinimumElevation")
        self.merrittGroundStation.maximumRange = self.initial_conditions.get("merrittMaximumRange")
        self.merrittGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create a Singapore ground station (NASA's Near Earth Network)
        self.singaporeGroundStation = groundLocation.GroundLocation()
        self.singaporeGroundStation.ModelTag = "GroundStation3"
        self.singaporeGroundStation.currentGroundStateOutMsgName = "GroundStation3_Location"
        self.singaporeGroundStation.planetRadius = self.initial_conditions.get("singaporeGroundStationPlanetRadius")
        self.singaporeGroundStation.specifyLocation(self.initial_conditions.get("singaporeGroundStationLat"),
                                                  self.initial_conditions.get("singaporeGroundStationLong"),
                                                  self.initial_conditions.get("singaporeGroundStationAlt"))
        self.singaporeGroundStation.planetInMsgName = planet.bodyInMsgName
        self.singaporeGroundStation.minimumElevation = self.initial_conditions.get("singaporeMinimumElevation")
        self.singaporeGroundStation.maximumRange = self.initial_conditions.get("singaporeMaximumRange")
        self.singaporeGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create a Weilheim Germany ground station (NASA's Near Earth Network)
        self.weilheimGroundStation = groundLocation.GroundLocation()
        self.weilheimGroundStation.ModelTag = "GroundStation4"
        self.weilheimGroundStation.currentGroundStateOutMsgName = "GroundStation4_Location"
        self.weilheimGroundStation.planetRadius = self.initial_conditions.get("weilheimGroundStationPlanetRadius")
        self.weilheimGroundStation.specifyLocation(self.initial_conditions.get("weilheimGroundStationLat"),
                                                    self.initial_conditions.get("weilheimGroundStationLong"),
                                                    self.initial_conditions.get("weilheimGroundStationAlt"))
        self.weilheimGroundStation.planetInMsgName = planet.bodyInMsgName
        self.weilheimGroundStation.minimumElevation = self.initial_conditions.get("weilheimMinimumElevation")
        self.weilheimGroundStation.maximumRange = self.initial_conditions.get("weilheimMaximumRange")
        self.weilheimGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create a Santiago, Chile ground station (NASA's Near Earth Network)
        self.santiagoGroundStation = groundLocation.GroundLocation()
        self.santiagoGroundStation.ModelTag = "GroundStation5"
        self.santiagoGroundStation.currentGroundStateOutMsgName = "GroundStation5_Location"
        self.santiagoGroundStation.planetRadius = self.initial_conditions.get("santiagoGroundStationPlanetRadius")
        self.santiagoGroundStation.specifyLocation(self.initial_conditions.get("santiagoGroundStationLat"),
                                                   self.initial_conditions.get("santiagoGroundStationLong"),
                                                   self.initial_conditions.get("santiagoGroundStationAlt"))
        self.santiagoGroundStation.planetInMsgName = planet.bodyInMsgName
        self.santiagoGroundStation.minimumElevation = self.initial_conditions.get("santiagoMinimumElevation")
        self.santiagoGroundStation.maximumRange = self.initial_conditions.get("santiagoMaximumRange")
        self.santiagoGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create a Dongara, Australia ground station (NASA's Near Earth Network)
        self.dongaraGroundStation = groundLocation.GroundLocation()
        self.dongaraGroundStation.ModelTag = "GroundStation6"
        self.dongaraGroundStation.currentGroundStateOutMsgName = "GroundStation6_Location"
        self.dongaraGroundStation.planetRadius = self.initial_conditions.get("dongaraGroundStationPlanetRadius")
        self.dongaraGroundStation.specifyLocation(self.initial_conditions.get("dongaraGroundStationLat"),
                                                  self.initial_conditions.get("dongaraGroundStationLong"),
                                                  self.initial_conditions.get("dongaraGroundStationAlt"))
        self.dongaraGroundStation.planetInMsgName = planet.bodyInMsgName
        self.dongaraGroundStation.minimumElevation = self.initial_conditions.get("dongaraMinimumElevation")
        self.dongaraGroundStation.maximumRange = self.initial_conditions.get("dongaraMaximumRange")
        self.dongaraGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create a Dongara, Australia ground station (NASA's Near Earth Network)
        self.hawaiiGroundStation = groundLocation.GroundLocation()
        self.hawaiiGroundStation.ModelTag = "GroundStation7"
        self.hawaiiGroundStation.currentGroundStateOutMsgName = "GroundStation7_Location"
        self.hawaiiGroundStation.planetRadius = self.initial_conditions.get("hawaiiGroundStationPlanetRadius")
        self.hawaiiGroundStation.specifyLocation(self.initial_conditions.get("hawaiiGroundStationLat"),
                                                  self.initial_conditions.get("hawaiiGroundStationLong"),
                                                  self.initial_conditions.get("hawaiiGroundStationAlt"))
        self.hawaiiGroundStation.planetInMsgName = planet.bodyInMsgName
        self.hawaiiGroundStation.minimumElevation = self.initial_conditions.get("hawaiiMinimumElevation")
        self.hawaiiGroundStation.maximumRange = self.initial_conditions.get("hawaiiMaximumRange")
        self.hawaiiGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsgName)

        # Create an instrument
        self.instrument = simpleInstrument.SimpleInstrument()
        self.instrument.ModelTag = "instrument" + str(sc_number)
        self.instrument.nodeBaudRate = self.initial_conditions.get("instrumentBaudRate") # baud
        self.instrument.nodeDataName = "Instrument" + str(sc_number)
        self.instrument.nodeDataOutMsgName = "InstrumentMsg" + str(sc_number)

        # Create a "transmitter"
        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.transmitter.ModelTag = "transmitter" + str(sc_number)
        self.transmitter.nodeBaudRate = self.initial_conditions.get("transmitterBaudRate")   # baud
        self.transmitter.packetSize = self.initial_conditions.get("transmitterPacketSize")  # bits
        self.transmitter.numBuffers = self.initial_conditions.get("transmitterNumBuffers")
        self.transmitter.nodeDataOutMsgName = "TransmitterMsg" + str(sc_number)
        self.transmitter.addAccessMsgToTransmitter(self.boulderGroundStation.accessOutMsgNames[-1])
        self.transmitter.addAccessMsgToTransmitter(self.merrittGroundStation.accessOutMsgNames[-1])
        self.transmitter.addAccessMsgToTransmitter(self.singaporeGroundStation.accessOutMsgNames[-1])
        self.transmitter.addAccessMsgToTransmitter(self.weilheimGroundStation.accessOutMsgNames[-1])
        self.transmitter.addAccessMsgToTransmitter(self.santiagoGroundStation.accessOutMsgNames[-1])
        self.transmitter.addAccessMsgToTransmitter(self.dongaraGroundStation.accessOutMsgNames[-1])
        self.transmitter.addAccessMsgToTransmitter(self.hawaiiGroundStation.accessOutMsgNames[-1])

        # Create a partitionedStorageUnit and attach the instrument to it
        self.storageUnit = simpleStorageUnit.SimpleStorageUnit()
        self.storageUnit.ModelTag = "storageUnit" + str(sc_number)
        self.storageUnit.storageUnitDataOutMsgName = "storageUnit" + str(sc_number)
        self.storageUnit.storageCapacity = self.initial_conditions.get("dataStorageCapacity") # bits (1 GB)
        self.storageUnit.addDataNodeToModel(self.instrument.nodeDataOutMsgName)
        self.storageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsgName)

        # Add the storage unit to the transmitter
        self.transmitter.addStorageUnitToTransmitter(self.storageUnit.storageUnitDataOutMsgName)

        # Initialize the observations (normed)
        # Inertial position
        self.obs[0, 0] = self.scObject.hub.r_CN_NInit[0]/np.linalg.norm(self.scObject.hub.r_CN_NInit)
        self.obs[1, 0] = self.scObject.hub.r_CN_NInit[1]/np.linalg.norm(self.scObject.hub.r_CN_NInit)
        self.obs[2, 0] = self.scObject.hub.r_CN_NInit[2]/np.linalg.norm(self.scObject.hub.r_CN_NInit)
        # Inertial velocity
        self.obs[3, 0] = self.scObject.hub.v_CN_NInit[0]/np.linalg.norm(self.scObject.hub.v_CN_NInit)
        self.obs[4, 0] = self.scObject.hub.v_CN_NInit[1]/np.linalg.norm(self.scObject.hub.v_CN_NInit)
        self.obs[5, 0] = self.scObject.hub.v_CN_NInit[2]/np.linalg.norm(self.scObject.hub.v_CN_NInit)
        # Attitude error
        self.obs[6, 0] = np.linalg.norm(self.scObject.hub.sigma_BNInit)
        # Attitude rate
        self.obs[7, 0] = np.linalg.norm(self.scObject.hub.omega_BN_BInit)
        # Wheel speeds
        self.obs[8, 0] = self.initial_conditions.get("wheelSpeeds")[0]/(mc.RPM*6000)
        self.obs[9, 0] = self.initial_conditions.get("wheelSpeeds")[1]/(mc.RPM*6000)
        self.obs[10, 0] = self.initial_conditions.get("wheelSpeeds")[2]/(mc.RPM*6000)
        # Stored charge
        self.obs[11, 0] = self.powerMonitor.storedCharge_Init/self.initial_conditions.get("batteryStorageCapacity")

        # Initialize the full observations
        # Inertial position
        self.obs_full[0:3, 0] = np.asarray(self.scObject.hub.r_CN_NInit).flatten()
        # Inertial velocity
        self.obs_full[3:6, 0] = np.asarray(self.scObject.hub.v_CN_NInit).flatten()
        # Attitude error
        self.obs_full[6, 0] = np.linalg.norm(self.scObject.hub.sigma_BNInit)
        # Attitude rate
        self.obs_full[7, 0] = np.linalg.norm(self.scObject.hub.omega_BN_BInit)
        # Wheel speeds
        self.obs_full[8:11, 0] = self.initial_conditions.get("wheelSpeeds")[0:3]*mc.RPM
        # Stored charge
        self.obs_full[11, 0] = self.powerMonitor.storedCharge_Init/3600.0

        # Eclipse indicator
        self.obs[12, 0] = self.obs_full[12, 0] = 0
        # Stored data
        self.obs[13, 0] = self.obs_full[13, 0] = 0
        # Transmitted data
        self.obs[14, 0] = self.obs_full[14, 0] = 0
        # Ground Station access indicators
        self.obs[15:22, 0] = self.obs_full[15:22, 0] = 0
        # Set the percentage through the planning interval
        self.obs[22] = self.obs_full[22] = 0

        self.obs = np.around(self.obs, decimals=5)

        # Add all the models to the tasks
        # Spice Task
        self.AddModelToTask(self.spiceTaskName, self.gravFactory.spiceObject, ModelPriority=1100)

        # Dyn Task
        self.AddModelToTask(self.dynTaskName, self.densityModel, ModelPriority=1000)
        self.AddModelToTask(self.dynTaskName, self.dragEffector, ModelPriority=999)
        self.AddModelToTask(self.dynTaskName, self.simpleNavObject, ModelPriority=998)
        self.AddModelToTask(self.dynTaskName, self.rwStateEffector, ModelPriority=997)
        self.AddModelToTask(self.dynTaskName, self.thrusterSet, ModelPriority=996)
        self.AddModelToTask(self.dynTaskName, self.scObject, ModelPriority=899)

        # Env Task
        self.AddModelToTask(self.envTaskName, self.boulderGroundStation, ModelPriority=995)
        self.AddModelToTask(self.envTaskName, self.merrittGroundStation, ModelPriority=994)
        self.AddModelToTask(self.envTaskName, self.singaporeGroundStation, ModelPriority=993)
        self.AddModelToTask(self.envTaskName, self.weilheimGroundStation, ModelPriority=992)
        self.AddModelToTask(self.envTaskName, self.santiagoGroundStation, ModelPriority=991)
        self.AddModelToTask(self.envTaskName, self.dongaraGroundStation, ModelPriority=990)
        self.AddModelToTask(self.envTaskName, self.hawaiiGroundStation, ModelPriority=989)
        self.AddModelToTask(self.envTaskName, self.eclipseObject, ModelPriority=988)
        self.AddModelToTask(self.envTaskName, self.solarPanel, ModelPriority=898)
        self.AddModelToTask(self.envTaskName, self.instrumentPowerSink, ModelPriority=897)
        self.AddModelToTask(self.envTaskName, self.transmitterPowerSink, ModelPriority=896)
        self.AddModelToTask(self.envTaskName, self.instrument, ModelPriority=895)
        self.AddModelToTask(self.envTaskName, self.powerMonitor, ModelPriority=799)
        self.AddModelToTask(self.envTaskName, self.transmitter, ModelPriority=798)
        self.AddModelToTask(self.envTaskName, self.storageUnit, ModelPriority=699)

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
        self.dynProc.addTask(self.CreateNewTask("sunPointTask", self.processTasksTimeStep),taskPriority=99)
        self.dynProc.addTask(self.CreateNewTask("nadirPointTask", self.processTasksTimeStep),taskPriority=98)
        self.dynProc.addTask(self.CreateNewTask("mrpControlTask", self.processTasksTimeStep), taskPriority=96)
        self.dynProc.addTask(self.CreateNewTask("rwDesatTask", self.processTasksTimeStep), taskPriority=97)

        #   Specify the vehicle configuration message to tell things what the vehicle inertia is
        vehicleConfigOut = fswMessages.VehicleConfigFswMsg()
        # use the same inertia in the FSW algorithm as in the simulation
        #   Set inertia properties to those of a solid 6U cubeoid:
        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")

        # Grab the mass for readability of inertia calc
        mass = self.initial_conditions.get("mass")

        I = [1. / 12. * mass * (width ** 2. + depth ** 2.), 0., 0.,
             0., 1. / 12. * mass * (depth ** 2. + height ** 2.), 0.,
             0., 0., 1. / 12. * mass * (width ** 2. + height ** 2.)]

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
        self.AddModelToTask("sunPointTask", self.sunPointWrap, self.sunPointData, ModelPriority=1200)

        self.AddModelToTask("nadirPointTask", self.hillPointWrap, self.hillPointData, ModelPriority=1199)

        self.AddModelToTask("mrpControlTask", self.mrpFeedbackControlWrap, self.mrpFeedbackControlData,
                            ModelPriority=1198)
        self.AddModelToTask("mrpControlTask", self.trackingErrorWrap, self.trackingErrorData, ModelPriority=1197)
        self.AddModelToTask("mrpControlTask", self.rwMotorTorqueWrap, self.rwMotorTorqueConfig, ModelPriority=1196)

        self.AddModelToTask("rwDesatTask", self.thrDesatControlWrap, self.thrDesatControlConfig, ModelPriority=1195)
        self.AddModelToTask("rwDesatTask", self.thrForceMappingWrap, self.thrForceMappingConfig, ModelPriority=1194)
        self.AddModelToTask("rwDesatTask", self.thrDumpWrap, self.thrDumpConfig, ModelPriority=1193)

    def set_logging(self):
        '''
        Logs simulation outputs to return as observations. This simulator observes:
        mrp_bn - inertial to body MRP
        error_mrp - Attitude error given current guidance objective
        power_level - current W-Hr from the battery
        r_bn - inertial position of the s/c relative to Earth
        :return:
        '''

        # Set the sampling time to the duration of a timestep:
        samplingTime = mc.sec2nano(self.dynRate)

        #   Log inertial attitude, position
        self.TotalSim.logThisMessage(self.scObject.scStateOutMsgName, samplingTime)

        # RW power
        for c in range(0,3):
            self.TotalSim.logThisMessage(self.rwPowerList[c].nodePowerOutMsgName, samplingTime)

        self.TotalSim.logThisMessage(self.simpleNavObject.outputTransName,
                                               samplingTime)
        self.TotalSim.logThisMessage(self.simpleNavObject.outputAttName,
                                     samplingTime)
        self.TotalSim.logThisMessage(self.rwStateEffector.OutputDataString,
                                     samplingTime)
        #   Log FSW error portrait
        self.TotalSim.logThisMessage(self.trackingErrorData.outputDataName, samplingTime)
        self.TotalSim.logThisMessage(self.mrpFeedbackControlData.outputDataName, samplingTime)

        #   Log system power status
        self.TotalSim.logThisMessage(self.powerMonitor.batPowerOutMsgName,
                                               samplingTime)

        #   Eclipse indicator
        self.TotalSim.logThisMessage(self.solarPanel.sunEclipseInMsgName, samplingTime)

        # Storage Unit, Transmitter, and Ground Station messages
        self.TotalSim.logThisMessage(self.boulderGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.singaporeGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.merrittGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.weilheimGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.santiagoGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.dongaraGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.hawaiiGroundStation.accessOutMsgNames[-1], samplingTime)
        self.TotalSim.logThisMessage(self.storageUnit.storageUnitDataOutMsgName, samplingTime)
        self.TotalSim.logThisMessage(self.transmitter.nodeDataOutMsgName, samplingTime)

        self.TotalSim.logThisMessage("earth_planet_data", samplingTime)

        self.TotalSim.logThisMessage(self.boulderGroundStation.currentGroundStateOutMsgName, samplingTime)

        return

    def run_sim(self, action, return_obs = True):
        '''
        Executes the sim for a specified duration given a mode command.
        :param action:
        :param duration:
        :return:
        '''

        # Turn mode request into a string
        self.modeRequest = str(action)
        # Set the sim_over param to false
        self.sim_over = False

        currentResetTime = mc.sec2nano(self.simTime)

        # Imaging mode
        if self.modeRequest == "0":
            self.dynProc.enableAllTasks()
            self.hillPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Disable sun pointing and desat
            self.disableTask('sunPointTask')
            self.disableTask('rwDesatTask')
            # Turn off transmitter
            self.transmitter.dataStatus = 0
            self.transmitterPowerSink.powerStatus = 0
            # Turn on instrument
            self.instrument.dataStatus = 1
            self.instrumentPowerSink.powerStatus = 1
            # Turn on nadir pointing and MRP control
            self.enableTask('nadirPointTask')
            self.enableTask('mrpControlTask')

        # Battery charging mode
        elif self.modeRequest == "1":
            self.dynProc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Turn off nadir pointing and desat
            self.disableTask('nadirPointTask')
            self.disableTask('rwDesatTask')
            # Turn off transmitter
            self.transmitter.dataStatus = 0
            self.transmitterPowerSink.powerStatus = 0
            # Turn off instrument
            self.instrument.dataStatus = 0
            self.instrumentPowerSink.powerStatus = 0
            # Turn on sun pointing and MRP control
            self.enableTask('sunPointTask')
            self.enableTask('mrpControlTask')

        # Desaturation mode
        elif self.modeRequest == "2":
            self.dynProc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Reset
            self.thrDesatControlWrap.Reset(currentResetTime)
            self.thrDumpWrap.Reset(currentResetTime)
            # Turn off nadir pointing and sun pointing
            self.disableTask('nadirPointTask')
            self.disableTask('sunPointTask')
            # Turn off transmitter
            self.transmitter.dataStatus = 0
            self.transmitterPowerSink.powerStatus = 0
            # Turn off instrument
            self.instrument.dataStatus = 0
            self.instrumentPowerSink.powerStatus = 0
            # Turn on sunPoint, MRP control, and desat
            self.enableTask('sunPointTask')
            self.enableTask('mrpControlTask')
            self.enableTask('rwDesatTask')

        # Downlink mode
        elif self.modeRequest == "3":
            self.dynProc.enableAllTasks()
            self.hillPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Disable sunPoint and Desat tasks
            self.disableTask('sunPointTask')
            self.disableTask('rwDesatTask')
            # Turn off instrument
            self.instrument.dataStatus = 0
            self.instrumentPowerSink.powerStatus = 0
            # Turn on transmitter
            self.transmitter.dataStatus = 1
            self.transmitterPowerSink.powerStatus = 1
            # Turn on nadir pointing and MRP control
            self.enableTask('nadirPointTask')
            self.enableTask('mrpControlTask')

        # Increment time and the current step
        self.simTime += self.step_duration
        simulation_time = mc.sec2nano(self.simTime)
        self.curr_step += 1

        #   Execute the sim
        self.ConfigureStopTime(simulation_time)
        self.ExecuteSimulation()

        if return_obs:
            # Pull observations for the final state in the mode
            simDict_single = self.pullMultiMessageLogData([
                self.scObject.scStateOutMsgName + '.r_BN_N',
                self.scObject.scStateOutMsgName + '.v_BN_N',
                self.simpleNavObject.outputAttName + '.omega_BN_B',
                self.trackingErrorData.outputDataName + '.sigma_BR',
                self.rwStateEffector.OutputDataString + '.wheelSpeeds',
                self.powerMonitor.batPowerOutMsgName + '.storageLevel',
                self.solarPanel.sunEclipseInMsgName + '.shadowFactor',
                self.storageUnit.storageUnitDataOutMsgName + '.storageLevel',
                "earth_planet_data.J20002Pfix",
                self.boulderGroundStation.currentGroundStateOutMsgName+'.r_LP_N'
            ], [list(range(3)), list(range(3)), list(range(3)), list(range(3)), list(range(3)), list(range(1)),
                list(range(1)), list(range(1)), list(range(9)), list(range(3))],
                ['double','double','double', 'double','double','double', 'double','double','double', 'double', 'double', 'double', 'double'], 1)

            # Pull the ground station access times and transmitter baudRate over the entire step
            simDict_full = self.pullMultiMessageLogData([
                self.boulderGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.merrittGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.singaporeGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.weilheimGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.santiagoGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.dongaraGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.hawaiiGroundStation.accessOutMsgNames[-1] + '.hasAccess',
                self.transmitter.nodeDataOutMsgName + '.baudRate'],
                [list(range(1)), list(range(1)), list(range(1)), list(range(1)), list(range(1)),
                 list(range(1)), list(range(1)), list(range(1))],
                ['int', 'int', 'int', 'int', 'int', 'int', 'int', 'double'],
                int(self.step_duration/self.dynRate))

            # Compute the relevant state variables
            attErr = simDict_single[self.trackingErrorData.outputDataName + '.sigma_BR']
            attRate = simDict_single[self.simpleNavObject.outputAttName + '.omega_BN_B']
            storedCharge = simDict_single[self.powerMonitor.batPowerOutMsgName + '.storageLevel']
            storedData = simDict_single[self.storageUnit.storageUnitDataOutMsgName + '.storageLevel']
            accessIndicator1 = simDict_full[self.boulderGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            accessIndicator2 = simDict_full[self.merrittGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            accessIndicator3 = simDict_full[self.singaporeGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            accessIndicator4 = simDict_full[self.weilheimGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            accessIndicator5 = simDict_full[self.santiagoGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            accessIndicator6 = simDict_full[self.dongaraGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            accessIndicator7 = simDict_full[self.hawaiiGroundStation.accessOutMsgNames[-1]+'.hasAccess']
            transmitterBaud = simDict_full[self.transmitter.nodeDataOutMsgName + '.baudRate']
            eclipseIndicator = simDict_single[self.solarPanel.sunEclipseInMsgName + '.shadowFactor']
            wheelSpeeds = simDict_single[self.rwStateEffector.OutputDataString+'.wheelSpeeds']

            # Get the rotation matrix from the inertial to planet frame from SPICE
            dcm_PN = np.array(simDict_single["earth_planet_data.J20002Pfix"][0][1:10]).reshape((3,3))

            # Get inertial position and velocity, rotate to planet-fixed frame
            inertialPos = simDict_single[self.scObject.scStateOutMsgName + '.r_BN_N']
            inertialVel = simDict_single[self.scObject.scStateOutMsgName + '.v_BN_N']
            planetFixedPos = np.matmul(dcm_PN, inertialPos[-1][1:4])
            planetFixedVel = np.matmul(dcm_PN, inertialVel[-1][1:4])

            # Full observations (non-normalized), all rounded to five decimals
            obs_full = np.hstack(np.around([planetFixedPos[0], planetFixedPos[1], planetFixedPos[2],
                planetFixedVel[0], planetFixedVel[1], planetFixedVel[2], np.linalg.norm(attErr[-1, 1:4]),
                np.linalg.norm(attRate[-1, 1:4]), wheelSpeeds[-1, 1], wheelSpeeds[-1, 2], wheelSpeeds[-1, 3],
                storedCharge[-1, 1]/3600., eclipseIndicator[-1, 1], storedData[-1, 1],
                np.sum(transmitterBaud[:, -1])*self.dynRate/8E6, np.sum(accessIndicator1[:, 1]),
                np.sum(accessIndicator2[:, 1])*self.dynRate, np.sum(accessIndicator3[:, 1])*self.dynRate,
                np.sum(accessIndicator4[:, 1])*self.dynRate, np.sum(accessIndicator5[:, 1])*self.dynRate,
                np.sum(accessIndicator6[:, 1])*self.dynRate, np.sum(accessIndicator7[:, 1])*self.dynRate,
                self.simTime/60], decimals=5))

            # Normalized observations, pull things from dictionary for readability
            transmitterBaudRate = self.initial_conditions.get("transmitterBaudRate")
            batteryStorageCapacity = self.initial_conditions.get("batteryStorageCapacity")
            dataStorageCapacity = self.initial_conditions.get("dataStorageCapacity")

            # Normalized observations, all rounded to five decimals
            obs_norm = np.hstack(np.around([planetFixedPos[0]/np.linalg.norm(planetFixedPos[0:3]),
                planetFixedPos[1]/np.linalg.norm(planetFixedPos[0:3]),
                planetFixedPos[2]/np.linalg.norm(planetFixedPos[0:3]),
                planetFixedVel[0]/np.linalg.norm(planetFixedVel[0:3]),
                planetFixedVel[1]/np.linalg.norm(planetFixedVel[0:3]),
                planetFixedVel[2]/np.linalg.norm(planetFixedVel[0:3]),
                np.linalg.norm(attErr[-1, 1:4]), np.linalg.norm(attRate[-1, 1:4]), wheelSpeeds[-1, 1]/(6000*mc.RPM),
                wheelSpeeds[-1, 2]/(6000*mc.RPM), wheelSpeeds[-1, 3]/(6000*mc.RPM),
                storedCharge[-1, 1]/batteryStorageCapacity, eclipseIndicator[-1, 1],
                storedData[-1, 1]/dataStorageCapacity,
                np.sum(transmitterBaud[:, -1])*self.dynRate/(transmitterBaudRate*self.step_duration),
                np.sum(accessIndicator1[:, 1])*self.dynRate/self.step_duration,
                np.sum(accessIndicator2[:, 1])*self.dynRate/self.step_duration,
                np.sum(accessIndicator3[:, 1])*self.dynRate/self.step_duration,
                np.sum(accessIndicator4[:, 1])*self.dynRate/self.step_duration,
                np.sum(accessIndicator5[:, 1])*self.dynRate/self.step_duration,
                np.sum(accessIndicator6[:, 1])*self.dynRate/self.step_duration,
                np.sum(accessIndicator7[:, 1])*self.dynRate/self.step_duration, self.simTime/(self.max_length*60)],
                                           decimals=5))

            # Reshape and assign the observations
            self.obs_full = obs_full.reshape(len(obs_full), 1)
            self.obs = obs_norm.reshape(len(obs_norm), 1)

            # Check if crashed into Earth
            if np.linalg.norm(inertialPos[-1,1:4]) < (orbitalMotion.REQ_EARTH/1000.):
                self.sim_over = True

        # return self.obs, self.sim_states, self.sim_over
        return self.obs, self.sim_over, self.obs_full

    def close_gracefully(self):
        """
        makes sure spice gets shut down right when we close.
        :return:
        """
        self.gravFactory.unloadSpiceKernels()
        # self.gravFactory.spiceObject.clearKeeper()
        # self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'de430.bsp')
        # self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'naif0012.tls')
        # self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'de-403-masses.tpc')
        # self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'pck00010.tpc')

        return

if __name__=="__main__":
    """
    Test execution of the simulator with random actions and plot the observation space.
    """
    sim = LEONadirSimulator(0.1, 1.0, 60.)
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
    plt.plot(range(0,tFinal), obs[:,3], label= "J_bat (W-Hr)")
    plt.plot(range(0,tFinal),obs[:,4], label="eclipse_ind")
    plt.legend()


    plt.figure()
    plt.plot(states[:, 3]/1000., states[:, 4]/1000., label="Orbit")
    plt.legend()

    plt.show()





