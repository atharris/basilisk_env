# 3rd party modules
import gym
import numpy as np
import scipy as sci
from scipy.linalg import expm
from gym import spaces

from gym_orbit.envs.utilities import StateLibrary as sl
from gym_orbit.envs.utilities import ActionLibrary as al
from gym_orbit.envs.utilities import orbitalMotion as om
from gym_orbit.envs.python_orbit.science_simpleObs import set_rand_default_ic
from basilisk_env.simulators import leoPowerAttitudeSimulator


class leoPowerAttEnv(gym.Env):
    """
    Simple attitude/orbit control problem. The spacecraft must decide when to point at the ground (which generates a
    reward) versus pointing at the sun (which increases the sim duration).
    """

    def __init__(self):
        self.__version__ = "0.0.1"
        print("Basilisk Attitude Mode Management Sim - Version {}".format(self.__version__))

        # General variables defining the environment
        self.max_length = 12*60 # Specify the maximum number of planning intervals

        # Create and store the simulation engine
        self.simulator = leoPowerAttitudeSimulator.LEOPowerAttitudeSimulator(0.1, 0.1, 60.0)  # Set taskrate equal to 0.1 seconds (10Hz)

        #   Set up options, constants for this environment
        self.step_duration = 60.  # Set step duration equal to 1 minute (180min ~ 2 orbits)
        self.reward_mult = 1.
        ## Observation space is:
        #  sigma_BN - 3 x 1 -  s/c attitude relative to the Earth
        #   r_BN - 3 x 1 - s/c position relative to the Earth
        #   v_BN - 3 x 1 - s/c velocity relative to the Earth
        #   W_stored - 1 x 1 - stored battery charge
        #   eclipse_ind - 1 x 1 1 in the sun, 0 in eclipse
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(low, high,shape=(5,1))

        self.obs = np.zeros([5,])

        ##  Action Space description
        #   0 - earth pointing (mission objective)
        #   1 - sun pointing (power objective)

        self.action_space = spaces.Discrete(2)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False

    def _seed(self):
        np.random.seed()
        self.simulator.powerMonitor.storedCharge_Init = np.clip(10*np.random.rand(1), 1.0, 10.0)[0]
        return

    def _step(self, action):
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
        if self.curr_step >= self.max_length:
            self.episode_over = True

        self._take_action(action)

        reward = self._get_reward()
        ob = self._get_state()

        if ob[3] == 0:
            self.episode_over = True

        info = {
            'full_states' : self.debug_states,
            'obs' : ob
        }

        self.curr_step += 1
        return ob, reward, self.episode_over, info

    def _take_action(self, action):
        '''
        Interfaces with the simulator to
        :param action:
        :return:
        '''

        self.action_episode_memory[self.curr_episode].append(action)

        #   Let the simulator handle action management:
        self.obs, self.debug_states = self.simulator.run_sim(action)

    def _get_reward(self):
        """
        Reward is based on time spent with the inertial attitude pointed towards the ground within a given tolerance.

        """
        reward_total = 0

        if self.action_episode_memory[self.curr_episode][-1] == 0:
            reward_total = self.reward_mult / (1. + np.linalg.norm(self.obs[0:3])**2.0)

        return reward_total

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.action_episode_memory.append([])
        self.episode_over = False
        self.curr_step = 0

        self.simulator = leoPowerAttitudeSimulator.LEOPowerAttitudeSimulator(.1, 0.1, 60.0)
        self.seed()

        return self.simulator.obs

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation.
        WIP: Work out which error representation to give the algo."""

        return self.simulator.obs