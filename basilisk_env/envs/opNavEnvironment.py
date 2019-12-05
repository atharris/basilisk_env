# 3rd party modules
import gym
import numpy as np
from gym import spaces

from basilisk_env.simulators import opNavSimulator



class opNavEnv(gym.Env):
    """
    OpNav scenario. The spacecraft must decide when to point at the ground (which generates a
    reward) versus pointing at the sun (which increases the sim duration).
    """

    def __init__(self):
        self.__version__ = "0.0.1"
        print("Basilisk OpNav Mode Management Sim - Version {}".format(self.__version__))

        # General variables defining the environment
        self.max_length =int(50) # Specify the maximum number of planning intervals

        #   Tell the environment that it doesn't have a sim attribute...
        self.sim_init = 0
        self.simulator = None
        self.reward_total = 0

        #   Set up options, constants for this environment
        self.step_duration = 60.  # Set step duration equal to 60 minute
        self.reward_mult = 1.
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(low, high,shape=(10,1))
        self.obs = np.zeros([10,])
        self.debug_states = np.zeros([9,])

        ##  Action Space description
        #   0 - earth pointing (mission objective)
        #   1 - sun pointing (power objective)
        #   2 - desaturation (required for long-term pointing)

        self.action_space = spaces.Discrete(2)

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
            self.simulator = opNavSimulator.scenario_OpNav(0.5, 0.5, self.step_duration)
            self.simulator_init = 1

        if self.curr_step >= self.max_length:
            self.episode_over = True

        prev_ob = self._get_state()
        self._take_action(action)

        reward = self._get_reward()
        self.reward_total += reward
        ob = self._get_state()

        if self.sim_over:
            self.episode_over = True
            print("End of episode")


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

        self.action_episode_memory[self.curr_episode].append(action)

        #   Let the simulator handle action management:
        self.obs, self.debug_states, self.sim_over = self.simulator.run_sim(action)

    def _get_reward(self):
        """
        Reward is based on time spent with the inertial attitude pointed towards the ground within a given tolerance.

        """
        reward = 0
        estErr = np.array([self.obs[3],self.obs[4], self.obs[5]])
        real = np.array([self.debug_states[0],self.debug_states[1], self.debug_states[2]])
        estErr -= real

        if self.action_episode_memory[self.curr_episode][-1] == 1:
            reward = np.linalg.norm(self.reward_mult / (1. + np.linalg.norm(estErr)**2.0))
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

        self.simulator = opNavSimulator.scenario_OpNav(0.5, 0.5, self.step_duration)
        self.simulator_init = 1
        self.seed()

        return self.simulator.obs

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation.
        WIP: Work out which error representation to give the algo."""

        return self.simulator.obs