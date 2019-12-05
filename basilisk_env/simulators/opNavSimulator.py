# 3rd party modules
import numpy as np

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from numpy.random import uniform
from Basilisk import __path__
bskPath = __path__[0]
# Get current file path
import sys, os, inspect, time, signal, subprocess
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
sys.path.append(path + '/opNav_models')
from BSK_masters import BSKSim, BSKScenario
import BSK_OpNavDynamics, BSK_OpNavFsw

class scenario_OpNav(BSKSim):
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
        super(scenario_OpNav, self).__init__(BSKSim)
        self.fswRate = fswRate
        self.dynRate = dynRate
        self.step_duration = step_duration

        self.set_DynModel(BSK_OpNavDynamics)
        self.set_FswModel(BSK_OpNavFsw)
        self.initInterfaces()
        self.filterUse = "relOD"
        self.configure_initial_conditions()
        self.get_DynModel().vizInterface.opNavMode = 2

        self.simTime = 0.0
        self.numModes = 50
        self.modeCounter = 0

        self.obs = np.zeros([10,1])
        self.sim_states = np.zeros([6,1])

        self.set_logging()
        # self.previousPointingGoal = "sunPointTask"

        self.modeRequest = 'OpNavOD'
        self.InitializeSimulationAndDiscover()

        return

    def configure_initial_conditions(self):
        # Configure Dynamics initial conditions
        oe = orbitalMotion.ClassicElements()
        oe.a = uniform(17000 * 1E3, 22000 * 1E3, 1)
        oe.e = uniform(0, 0.6, 1)
        oe.i = uniform(-20 * mc.D2R, 20 * mc.D2R, 1)
        oe.Omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
        oe.omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
        oe.f = uniform(0 * mc.D2R, 360 * mc.D2R, 1)

        # oe.a = 18000 * 1E3  # meters
        # oe.e = 0.6
        # oe.i = 10 * mc.D2R
        # oe.Omega = 25. * mc.D2R
        # oe.omega = 190. * mc.D2R
        # oe.f = 80. * mc.D2R  # 90 good
        mu = self.get_DynModel().marsGravBody.mu

        rN, vN = orbitalMotion.elem2rv(mu, oe)
        orbitalMotion.rv2elem(mu, rN, vN)

        rError = np.array([10000., 10000., -10000])
        vError = np.array([100, -10, 10])
        MRP = [0, 0, 0]
        self.get_FswModel().relativeODData.stateInit = (rN + rError).tolist() + (vN + vError).tolist()
        self.get_DynModel().scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)  # m   - r_CN_N
        self.get_DynModel().scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)  # m/s - v_CN_N
        self.get_DynModel().scObject.hub.sigma_BNInit = [[MRP[0]], [MRP[1]], [MRP[2]]]  # sigma_BN_B
        self.get_DynModel().scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]  # rad/s - omega_BN_B

        qNoiseIn = np.identity(6)
        qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 1E-3 * 1E-3
        qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6] * 1E-4 * 1E-4
        self.get_FswModel().relativeODData.qNoise = qNoiseIn.reshape(36).tolist()
        self.get_FswModel().imageProcessing.noiseSF = 1
        self.get_FswModel().relativeODData.noiseSF = 5


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
        samplingTime = mc.sec2nano(self.dynRate)

        #   Log planet, sun positions
        self.TotalSim.logThisMessage(self.get_FswModel().relativeODData.filtDataOutMsgName, samplingTime)
        self.TotalSim.logThisMessage(self.get_FswModel().imageProcessing.opnavCirclesOutMsgName, samplingTime)
        self.TotalSim.logThisMessage(self.get_DynModel().scObject.scStateOutMsgName, samplingTime)

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
        self.modeCounter+=1

        currentResetTime = mc.sec2nano(self.simTime)
        if self.modeRequest == "0":
            self.get_DynModel().cameraMod.cameraIsOn = 1
            self.modeRequest = 'OpNavOD'

        elif self.modeRequest == "1":
            self.get_DynModel().cameraMod.cameraIsOn = 0
            self.modeRequest = "sunSafePoint"

        self.simTime += self.step_duration
        simulationTime = mc.min2nano(self.simTime)

        #   Execute the sim
        self.ConfigureStopTime(simulationTime)
        self.ExecuteSimulation()

        NUM_STATES = 6
        #   Pull logged message data and return it as an observation
        simDict = self.pullMultiMessageLogData([
            self.get_DynModel().scObject.scStateOutMsgName + '.r_BN_N',
            self.get_DynModel().scObject.scStateOutMsgName + '.v_BN_N',
            self.get_DynModel().scObject.scStateOutMsgName + '.sigma_BN',
            self.get_FswModel().imageProcessing.opnavCirclesOutMsgName+ ".valid",
            self.get_FswModel().relativeODData.filtDataOutMsgName + ".state",
            self.get_FswModel().relativeODData.filtDataOutMsgName + ".covar"
        ], [list(range(3)), list(range(3)), list(range(3)), list(range(1)), list(range(NUM_STATES)), list(range(NUM_STATES*NUM_STATES))], 1)

        validCircle = simDict[self.get_FswModel().imageProcessing.opnavCirclesOutMsgName + ".valid"]
        position_N = simDict[self.get_DynModel().scObject.scStateOutMsgName + '.r_BN_N']
        sigma_BN = simDict[self.get_DynModel().scObject.scStateOutMsgName + '.sigma_BN']
        velocity_N = simDict[self.get_DynModel().scObject.scStateOutMsgName + '.v_BN_N']
        navState = simDict[self.get_FswModel().relativeODData.filtDataOutMsgName + ".state"]
        navCovar = simDict[self.get_FswModel().relativeODData.filtDataOutMsgName + ".covar"]

        mu = self.get_DynModel().marsGravBody.mu
        oe = orbitalMotion.rv2elem_parab(mu, navState[-1,1:4], navState[-1,4:7])
        covarVec = np.array([np.sqrt(navCovar[-1,1]), np.sqrt(navCovar[-1,2 + NUM_STATES]), np.sqrt(navCovar[-1,3 + 2*NUM_STATES])])

        debug = np.hstack([position_N[-1,1:4], velocity_N[-1,1:4], sigma_BN[-1,1:4]])
        obs = np.hstack([oe.a, oe.e, oe.f, navState[-1,1:4]/np.linalg.norm(navState[-1,1:4]), covarVec/np.linalg.norm(navState[-1,1:4]), validCircle[-1,1]])
        self.obs = obs.reshape(len(obs), 1)
        self.sim_states = debug.reshape(len(debug), 1)

        if self.modeCounter >= self.numModes:
            self.sim_over = True

        return self.obs, self.sim_states, self.sim_over

    def close_gracefully(self):
        """
        makes sure spice gets shut down right when we close.
        :return:
        """
        self.gravFactory.unloadSpiceKernels()
        return

def create_scenario_OpNav():
    return scenario_OpNav(0.5, 0.5, 60.)

if __name__=="__main__":
    """
    Test execution of the simulator with random actions and plot the observation space.
    """

    appPath = '/Applications/OpNavScene.app'
    # appPath = '/Applications/Vizard.app'
    child = subprocess.Popen(["open", appPath, "--args", "-opNavMode", "tcp://localhost:5556"])  # ,, "-batchmode"

    sim = scenario_OpNav(0.5, 0.5, 60.)
    obs = []
    states = []
    normWheelSpeed = []
    actList = []
    from matplotlib import pyplot as plt
    from random import randrange

    tFinal = 5
    for ind in range(0,tFinal):
        act = randrange(2)
        actList.append(act)
        ob, state, _ = sim.run_sim(act)
        #normWheelSpeed.append(np.linalg.norm(abs(ob[3:6])))
        obs.append(ob)
        states.append(state)
    obs = np.asarray(obs)
    states = np.asarray(states)

    plt.figure()
    plt.plot(range(0,tFinal),obs[:,0], label="a")
    plt.plot(range(0,tFinal),obs[:,1], label="e")
    plt.plot(range(0,tFinal),obs[:,2], label="f")
    plt.plot(range(0,tFinal),obs[:,9], label="val")
    plt.legend()

    plt.figure()
    plt.plot(range(0,tFinal),obs[:,3], label="pos1")
    plt.plot(range(0,tFinal),obs[:,4], label="pos2")
    plt.plot(range(0,tFinal),obs[:,5], label="pos3")
    plt.plot(range(0,tFinal),obs[:,6], label="cov1")
    plt.plot(range(0,tFinal),obs[:,7], label="cov2")
    plt.plot(range(0,tFinal),obs[:,8], label="cov3")
    plt.legend()


    plt.figure()
    plt.plot(range(0,tFinal),states[:,6], label="sigma1")
    plt.plot(range(0,tFinal),states[:,7], label="sigma2")
    plt.plot(range(0,tFinal),states[:,8], label="sigma3")
    plt.legend()

    plt.figure()
    plt.plot(states[:, 3]/1000., states[:, 4]/1000., label="Orbit")
    #plt.plot(states[:,12]/1000., states[:,13]/1000, label="Sun Position")
    plt.legend()

    plt.show()





