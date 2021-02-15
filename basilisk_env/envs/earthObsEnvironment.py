# 3rd party modules
import gym
import numpy as np
import scipy as sci
from scipy.linalg import expm
from gym import spaces
import copy

from basilisk_env.simulators import earthObsSimulator
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion as om


class EarthObsEnvironment(gym.Env):
    """
    Earth observation environment - simulates a spacecraft with dual imager modes attempting to image a ground location. 
    Agent must chose between imaging, charging, state management; also needs to chose an appropriate imaging type.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("Earth Observation Sim - Version {}".format(self.__version__))

        # General variables defining the environment
        self.max_length =int(3*180) # Specify the maximum number of planning intervals

        #   Tell the environment that it doesn't have a sim attribute...
        self.simulator_init = 0
        self.simulator = None
        self.simulator_backup = None
        self.reward_total = 0

        #   Set some attributes for the simulator; parameterized such that they can be varied in-sim
        self.mass = 330.0 # kg
        self.powerDraw = -5. #  W
        self.wheel_limit = 3000*mc.RPM # 3000 RPM in radians/s
        self.power_max = 20.0 # W/Hr

        #   Set up options, constants for this environment
        self.step_duration = 180.  # Set step duration equal to 1 minute (180min ~ 2 orbits)
        self.reward_mult = 1./self.max_length # Normalize reward to episode duration; '1' represents 100% ground observation time
        self.failure_penalty = 1 #    Default is 50.
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(low, high,shape=(14,1))
        self.obs = np.zeros([14,])

        ##  Action Space description
        #   0 - Visible imaging
        #   1 - IR Imaging
        #   2 - sun pointing (power objective)
        #   3 - desaturation (required for long-term pointing)

        self.action_space = spaces.Discrete(3)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False

    def _seed(self):
        np.random.seed()
        return

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        if self.simulator_init == 0:
            self.simulator = earthObsSimulator.EarthObsSimulator(.1, 1.0, self.step_duration, mass=self.mass, powerDraw = self.powerDraw)
            self.simulator_init = 1

        if self.curr_step >= self.max_length:
            self.episode_over = True

        prev_ob = self._get_state()
        self._take_action(action)

        reward = self._get_reward()
        self.reward_total += reward
        ob = self._get_state()
        ob[8] = ob[8] / self.wheel_limit #    Normalize reaction wheel speed to fraction of limit
        ob[9] = ob[9] / self.power_max #    Normalize current power to fraction of total power
        #   If the wheel speeds get too large, end the episode.
        if ob[8] > 1:
            self.episode_over = True
            reward -= self.failure_penalty
            self.reward_total -= self.failure_penalty
            print("Died from wheel explosion. RPMs were norm:"+str(ob[2]*self.wheel_limit)+", limit is "+str(self.wheel_limit)+", body rate was "+str(ob[1])+"action taken was "+str(action)+", env step"+str(self.curr_step))
            print("Prior state was RPM:"+str(prev_ob[8]*self.wheel_limit)+" . body rate was:"+str(prev_ob[1]))

        #   If we run out of power, end the episode.
        if ob[9] == 0:
            self.episode_over = True
            reward -= self.failure_penalty
            self.reward_total -= self.failure_penalty
            print("Ran out of power. Battery level was at at:"+str(prev_ob[9])+", env step"+str(self.curr_step-1))

        if self.sim_over:
            self.episode_over = True
            print("Orbit decayed - no penalty, but this one is over.")


        if self.episode_over:
            info = {'episode':{
                'r': self.reward_total,
                'l': self.curr_step},
                'full_states': self.debug_states,
                'obs': ob
            }
            self.simulator.close_gracefully() # Stop spice from blowing up
        else:
            info={
                'full_states': self.debug_states,
                'obs': ob
            }

        self.curr_step += 1
        return ob, reward, self.episode_over, info

    def _take_action(self, action):
        '''
        Interfaces with the simulator to
        :param action:
        :return:
        '''

        # print(self.curr_episode)

        self.action_episode_memory[self.curr_episode].append(action)

        #   Let the simulator handle action management:
        self.obs, self.debug_states, self.sim_over = self.simulator.run_sim(action)

    def _get_reward(self):
        """
        Reward is based on time spent with the inertial attitude pointed towards the ground within a given tolerance.

        """
        reward = 0
        last_action = self.action_episode_memory[self.curr_episode][-1]
        if (last_action == 0 or last_action==1) and self.obs[11,0]:
            #   Attitude contribution:
            att_reward = np.linalg.norm(self.reward_mult / (1. + self.obs[6,0]**2.0))
            #   Frequency contribution:
            time_since_last = self.obs[12+last_action,0]
            freq_mult = (time_since_last / self.max_length) +1.
             
            reward = att_reward * freq_mult
        return reward

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.action_episode_memory.append([])
        self.episode_over = False
        self.curr_step = 0
        self.reward_total = 0
        del(self.simulator) #   Force delete the sim to make sure nothing funky happens under the hood
        self.simulator = earthObsSimulator.EarthObsSimulator(.1, 1.0, self.step_duration)
        self.simulator_init = 1
        self.seed()
        ob = copy.deepcopy(self.simulator.obs)
        ob[8] = ob[8] / self.wheel_limit #    Normalize reaction wheel speed to fraction of limit
        ob[9] = ob[9] / self.power_max #    Normalize current power to fraction of total power
        return ob

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation.
        WIP: Work out which error representation to give the algo."""

        return self.simulator.obs

    def reset_init(self):
        self.action_episode_memory.append([])
        self.episode_over = False
        self.curr_step = 0
        self.reward_total = 0

        initial_conditions = self.simulator.initial_conditions
        del(self.simulator)
        self.simulator = earthObsSimulator.EarthObsSimulator(.1, 1.0, self.step_duration, initial_conditions)

        self.simulator_init = 1
        ob = copy.deepcopy(self.simulator.obs)
        ob[8] = ob[8] / self.wheel_limit #    Normalize reaction wheel speed to fraction of limit
        ob[9] = ob[9] / self.power_max #    Normalize current power to fraction of total power
        return ob

if __name__=="__main__":
    env = gym.make('earth_obs_env-v0')
    hist_list = []
    rhist_list = []
    #   Loop through the env twice
    for ind in range(0,2):
        hist = np.zeros([14,2*env.max_length])  
        r_hist = np.zeros([2*env.max_length])
        env.reset()
        env.seed(seed=12345)
        for step in range(0,env.max_length):
            ob, reward, ep_over, info = env.step(0)
            hist[:,step] = ob[:,0]
            r_hist[step] = reward
            if ep_over:
                break
        hist_list.append(hist)
        rhist_list.append(r_hist)
    tFinal = 2*env.max_length
    from matplotlib import pyplot as plt
    from matplotlib import rc 
    rc("text",usetex=False)
    for count,obs in enumerate(hist_list):
        rhist = rhist_list[count]
        plt.figure()
    
        plt.plot(range(0,tFinal),obs[0,:], label="r_lp_1")
        plt.plot(range(0,tFinal),obs[1,:], label="r_lp_2")
        plt.plot(range(0,tFinal),obs[2,:], label="r_lp_3")
        plt.plot(range(0,tFinal),obs[3,:], label="r_sc_1")
        plt.plot(range(0,tFinal),obs[4,:], label="r_sc_2")
        plt.plot(range(0,tFinal),obs[5,:], label="r_sc_3")
        plt.grid() 
        plt.legend()

        #   Plot on-board states
        plt.figure()
        plt.plot(range(0,tFinal), obs[6,:], label= "sigma_BR")
        plt.plot(range(0,tFinal),obs[7,:], label="omega_BN")
        plt.plot(range(0,tFinal),obs[8,:], label="omega_RW")
        plt.plot(range(0,tFinal),obs[9,:], label="J_bat")
        plt.plot(range(0,tFinal),obs[10,:], label="eclipse")
        plt.legend()
        plt.grid()

        #   Plot timeSinceAccess, modes, access indicators
        plt.figure()
        plt.plot(range(0,tFinal),obs[11,:], label="hadAccess")
        plt.plot(range(0,tFinal),obs[12,:], label="t_since_vis")
        plt.plot(range(0,tFinal),obs[13,:], label="t_since_ir")
        plt.grid()
        plt.legend()
        plt.title(f'Reward states for run {count}')

        plt.figure()
        print(f"Run {count} maximum reward: {max(rhist)}")
        plt.plot(range(0,tFinal), rhist,label='Reward')
        plt.grid()
        plt.legend()
        plt.title(f"Reward value for run {count}")
    plt.show()