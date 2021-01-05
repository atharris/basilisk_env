# 3rd party modules
import numpy as np

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk
from numpy.random import uniform
import matplotlib as mpl
from Basilisk import __path__
bskPath = __path__[0]
# Get current file path
import sys, os, inspect, time, signal, subprocess
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
sys.path.append(path + '/opNav_models')
from BSK_masters import BSKSim, BSKScenario
import BSK_OpNavDynamics, BSK_OpNavFsw

mpl.rcParams.update({'font.size' : 8 })
#seaborn-colorblind, 'seaborn-paper', 'bmh', 'tableau-colorblind10', 'seaborn-deep', 'myStyle', 'aiaa'

try:
    plt.style.use("myStyle")
except:
    pass
params = {'axes.labelsize': 8,'axes.titlesize':8, 'legend.fontsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'text.usetex': True}
mpl.rcParams.update(params)

class viz_manager(object):
    """
    Manages an instance of the Vizard astrodynamics visualizer.
    """
    def __init__(self):

        self.viz_init = False

    def createViz(self, port=5000):
        """
        Creates a Viz instance.
        """
        if self.viz_init == False:
            self.viz_proc = subprocess.Popen([os.environ['viz_app'], "--args", "-opNavMode", f"tcp://{os.environ['viz_address']}:{os.environ['viz_port']}"], stdout=subprocess.DEVNULL)  # ,, "-batchmode"
            self.viz_init=True
        else: 
            pass


    def stopViz(self):
        """ 
        Kills all existing vizard instances.
        """
        self.viz_proc.kill() #  Kill the viz process
        #   If we're on WSL, also invoke kill:
        proc = subprocess.Popen(["/mnt/c/Windows/System32/TASKKILL.exe","/IM","Vizard.exe", "/F"])
        time.sleep(1) # Give taskill some time to kill the task
        self.viz_init = False
        return 

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
        self.get_DynModel().vizInterface.opNavMode = 1

        self.viz_manager = viz_manager()
        self.viz_manager.createViz(port=5000)
        

        self.simTime = 0.0
        self.numModes = 50
        self.modeCounter = 0

        self.obs = np.zeros([4,1])
        self.sim_states = np.zeros([12,1])

        self.set_logging()
        # self.previousPointingGoal = "sunPointTask"

        self.modeRequest = 'OpNavOD'
        self.InitializeSimulationAndDiscover()

        return

    def configure_initial_conditions(self):
        # Configure Dynamics initial conditions
        oe = orbitalMotion.ClassicElements()
        # oe.a = uniform(17000 * 1E3, 22000 * 1E3, 1)
        # oe.e = uniform(0, 0.6, 1)
        # oe.i = uniform(-20 * mc.D2R, 20 * mc.D2R, 1)
        # oe.Omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
        # oe.omega = uniform(0 * mc.D2R, 360 * mc.D2R, 1)
        # oe.f = uniform(0 * mc.D2R, 360 * mc.D2R, 1)

        oe.a = 18000 * 1E3  # meters
        oe.e = 0.6
        oe.i = 10 * mc.D2R
        oe.Omega = 25. * mc.D2R
        oe.omega = 190. * mc.D2R
        oe.f = 80. * mc.D2R  # 90 good
        mu = self.get_DynModel().marsGravBody.mu

        rN, vN = orbitalMotion.elem2rv(mu, oe)
        orbitalMotion.rv2elem(mu, rN, vN)

        print("ICs \n")
        print(rN , "\n")
        print(vN , "\n")

        rError = uniform(100000,-100000, 3)
        vError = uniform(1000,-1000, 3)
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
        self.TotalSim.logThisMessage(self.get_DynModel().SimpleNavObject.outputAttName, samplingTime)
        self.TotalSim.logThisMessage(self.get_DynModel().scObject.scStateOutMsgName, samplingTime)

        return

    def run_sim(self, action):
        '''
        Executes the sim for a specified duration given a mode command.
        :param action:
        :param duration:
        :return:
        '''

        # self.modeRequest = str(action)

        self.sim_over = False
        self.modeCounter+=1

        currentResetTime = mc.sec2nano(self.simTime)
        if str(action) == "0":
            # self.get_DynModel().cameraMod.cameraIsOn = 1
            # self.modeRequest = 'OpNavOD'

            self.fswProc.disableAllTasks()
            self.enableTask('opNavPointTaskCheat')
            self.enableTask('mrpFeedbackRWsTask')
            self.enableTask('opNavODTask')


        elif str(action) == "1":
            self.get_DynModel().cameraMod.cameraIsOn = 0
            # self.modeRequest = "sunSafePoint"
            self.fswProc.disableAllTasks()
            self.enableTask('sunSafePointTask')
            self.enableTask('mrpFeedbackRWsTask')

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
            self.get_DynModel().SimpleNavObject.outputAttName + '.vehSunPntBdy',
            self.get_FswModel().relativeODData.filtDataOutMsgName + ".state",
            self.get_FswModel().relativeODData.filtDataOutMsgName + ".covar"
        ], [list(range(3)), list(range(3)), list(range(3)), list(range(3)), list(range(NUM_STATES)), list(range(NUM_STATES*NUM_STATES))], ['float']*6,numRecords=1)

        # sunHead = simDict["sun_planet_data" + ".PositionVector"]
        sunHead_B = simDict[self.get_DynModel().SimpleNavObject.outputAttName + '.vehSunPntBdy']
        position_N = simDict[self.get_DynModel().scObject.scStateOutMsgName + '.r_BN_N']
        sigma_BN = simDict[self.get_DynModel().scObject.scStateOutMsgName + '.sigma_BN']
        velocity_N = simDict[self.get_DynModel().scObject.scStateOutMsgName + '.v_BN_N']
        navState = simDict[self.get_FswModel().relativeODData.filtDataOutMsgName + ".state"]
        navCovar = simDict[self.get_FswModel().relativeODData.filtDataOutMsgName + ".covar"]

        # mu = self.get_DynModel().marsGravBody.mu
        # oe = orbitalMotion.rv2elem_parab(mu, navState[-1,1:4], navState[-1,4:7])
        covarVec = np.array([np.sqrt(navCovar[-1,1]), np.sqrt(navCovar[-1,2 + NUM_STATES]), np.sqrt(navCovar[-1,3 + 2*NUM_STATES])])

        BN = rbk.MRP2C(sigma_BN[-1,1:4])
        pos_B = -np.dot(BN, navState[-1,1:4]/np.linalg.norm(navState[-1,1:4]))
        sunHeadNorm = sunHead_B[-1,1:4]/np.linalg.norm(sunHead_B[-1,1:4])
        sunMarsAngle = np.dot(pos_B, sunHeadNorm)

        debug = np.hstack([navState[-1,1:4], position_N[-1,1:4], velocity_N[-1,1:4], sigma_BN[-1,1:4]])
        obs = np.hstack([sunMarsAngle, covarVec/np.linalg.norm(navState[-1,1:4])])
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
        self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'de430.bsp')
        self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'naif0012.tls')
        self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'de-403-masses.tpc')
        self.get_DynModel().SpiceObject.unloadSpiceKernel(self.get_DynModel().SpiceObject.SPICEDataPath, 'pck00010.tpc')

        self.viz_manager.stopViz()
        # try:
        #     os.kill(self.child.pid + 1, signal.SIGKILL)
        #     print("Closing Vizard")
        # except:
        #     print("IDK how to turn this thing off")
        return

def create_scenario_OpNav():
    return scenario_OpNav(1., 5., 50.)

if __name__=="__main__":
    """
    Test execution of the simulator with random actions and plot the observation space.
    """

    appPath = os.environ['viz_app']
    child = subprocess.Popen(
        [os.environ['viz_app'], "--args", "-opNavMode", f"tcp://{os.environ['viz_address']}:{os.environ['viz_port']}"],
         stdout=subprocess.DEVNULL)
         
    actHist = [1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    sim = scenario_OpNav(0.5, 5., 50.)
    obs = []
    states = []
    rewardList = []
    normWheelSpeed = []
    actList = []
    from matplotlib import pyplot as plt
    from random import randrange

    tFinal = len(actHist)
    rewardList.append(np.nan)
    for ind in range(0,len(actHist)):
        act = actHist[ind]#(ind-1)%2 #randrange(2)
        print("act = ",act)
        actList.append(act)
        ob, state, _ = sim.run_sim(act)
        reward = 0
        if act == 1:
            real = np.array([state[3],state[4], state[5]])
            nav = np.array([state[0],state[1], state[2]])
            nav -= real
            nav *= 1./np.linalg.norm(real)
            reward = np.linalg.norm(1./ (1. + np.linalg.norm(nav)**2.0))
        rewardList.append(reward)

        obs.append(ob)
        states.append(state)
    obs = np.asarray(obs)
    states = np.asarray(states)

    try:
        child.terminate()
    except:
        print("os.kill failed; fear for your lives")


    """Check results here"""
    colorsInt = len(mpl.cm.get_cmap("inferno").colors)/(10)
    colorList = []
    for i in range(10):
        colorList.append(mpl.cm.get_cmap("inferno").colors[int(i*colorsInt)])

    totalReward = [0]
    for ind in range(1, tFinal+1):
        totalReward.append(totalReward[-1] + rewardList[ind])

    plt.figure(num=22, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(rewardList, label='Mode-wise Reward', color=colorList[3])
    plt.plot(totalReward, label='Summed Reward', color=colorList[7])
    ax = plt.gca()
    for ind in range(0, tFinal):
        if actHist[ind] == 0:
            ax.axvspan(ind, ind + 1, color=colorList[1], alpha=0.05)
        if actHist[ind] == 1:
            ax.axvspan(ind, ind + 1, color=colorList[8], alpha=0.05)
    plt.ylabel('Reward')
    plt.xlabel('Modes (-)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('OpNav_Reward_History.pdf')

    nav = np.zeros(len(states[:,0]))
    covar = np.zeros(len(states[:,0]))

    nav1 = np.copy(states[:,0])
    nav1 -= states[:,3]
    nav2 = np.copy(states[:, 1])
    nav2 -= states[:, 4]
    nav3 = np.copy(states[:, 2])
    nav3 -= states[:, 5]

    for i in range(len(states[:,0])):
        nav[i] = np.linalg.norm(np.array([nav1[i],nav2[i],nav3[i]]))/ np.linalg.norm(np.array([nav1[0],nav2[0],nav3[0]]))
        covar[i] = np.linalg.norm(np.array([obs[i, 1],obs[i, 2],obs[i, 3]]))/np.linalg.norm(np.array([obs[0, 1],obs[0, 2],obs[0, 3]]))

    plt.figure(num=1, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(range(1,tFinal+1), nav, label="$\hat{\mathbf{r}}$", color = colorList[3])
    plt.plot(range(1,tFinal+1), covar, label="$\hat{\mathrm{P}}$", color = colorList[8])
    ax = plt.gca()
    for ind in range(0, tFinal):
        if actHist[ind] == 0:
            ax.axvspan(ind, ind + 1, color=colorList[1], alpha=0.05)
        if actHist[ind] == 1:
            ax.axvspan(ind, ind + 1, color=colorList[8], alpha=0.05)
    plt.legend(loc='best')
    plt.ylabel('Normalized States')
    plt.xlabel('Modes (-)')
    plt.savefig('ActionsNav.pdf')

    plt.figure(num=11, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(range(1,tFinal+1),obs[:,0], label="Angle")
    plt.plot(range(1,tFinal+1),covar, label="$\hat{\mathrm{P}}$")
    ax = plt.gca()
    for ind in range(0, tFinal):
        if actHist[ind] == 0:
            ax.axvspan(ind, ind + 1, color=colorList[1], alpha=0.05)
        if actHist[ind] == 1:
            ax.axvspan(ind, ind + 1, color=colorList[8], alpha=0.05)
    plt.legend(loc='best')
    plt.ylabel('Eclipse Angle vs Covar')
    plt.xlabel('Modes (-)')
    plt.savefig('AngleCovar.pdf')
    # plt.figure()
    # plt.plot(range(0,tFinal),nav1, label="nav1", color = 'r')
    # plt.plot(range(0,tFinal), nav2, label="nav2", color = 'g')
    # plt.plot(range(0,tFinal),nav3, label="nav3", color = 'b')
    # plt.plot(range(0, tFinal), obs[:, 1], label="cov1", color = 'r')
    # plt.plot(range(0, tFinal), obs[:, 2], label="cov2", color = 'g')
    # plt.plot(range(0, tFinal), obs[:, 3], label="cov3", color = 'b')
    # plt.legend()


    plt.figure(num=2, figsize=(2.7, 1.6), facecolor='w', edgecolor='k')
    plt.plot(range(1,tFinal+1),states[:,9], label="$\sigma_1$",  color = colorList[5])
    plt.plot(range(1,tFinal+1),states[:,10], label="$\sigma_2$", color = colorList[6])
    plt.plot(range(1,tFinal+1),states[:,11], label="$\sigma_3$", color = colorList[7])
    ax = plt.gca()
    for ind in range(0, tFinal):
        if actHist[ind] == 0:
            ax.axvspan(ind, ind + 1, color=colorList[1], alpha=0.05)
        if actHist[ind] == 1:
            ax.axvspan(ind, ind + 1, color=colorList[8], alpha=0.05)
    plt.legend(loc='best')
    plt.ylabel('Attitude')
    plt.xlabel('Modes (-)')
    plt.savefig('AttModes.pdf')


    plt.show()





