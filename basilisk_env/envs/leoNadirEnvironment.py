# 3rd party modules
import gym
import numpy as np
import scipy as sci
from scipy.linalg import expm
from gym import spaces
import copy

from basilisk_env.simulators import leoNadirSimulator
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion as om


class leoNadirEnv(gym.Env):
    """
    Extension of the leoPowerAttitudeEnvironment. The spacecraft must decide between pointing at the ground to collect
    science data, pointing at the sun to charge, desaturating reaction wheels, and downlinking data. This is referred to
    as the "nadir" simulator as science data is simply collected by nadir pointing. Specific imaging targets are not
    considered.
    """

    def __init__(self):
        self.__version__ = "0.0.1"
        print("Basilisk Attitude Mode Management Sim - Version {}".format(self.__version__))

        # General variables defining the environment
        self.max_length = int(3*180) # Specify the maximum number of planning intervals

        #   Tell the environment that it doesn't have a sim attribute...
        self.simulator_init = 0
        self.simulator = None
        self.reward_total = 0

        # Set initial conditions to none (gets assigned in reset)
        self.initial_conditions = None

        #   Set some attributes for the simulator; parameterized such that they can be varied in-sim
        self.mass = 330.0 # kg
        self.powerDraw = -5. #  W

        #   Set up options, constants for this environment
        self.step_duration = 120.  # seconds, tune as desired
        self.reward_mult = 0.95
        # self.failure_penalty = 1000 #    Default is 50.
        self.failure_penalty = 0.0 #    Default is 50.
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(low, high,shape=(5,1))
        self.obs = np.zeros([14,1])

        ##  Action Space description
        #   0 - Earth Pointing (point at Earth to take images)
        #   1 - sun pointing (point at the sun to charge battery)
        #   2 - desaturation (point at the sun and desaturate reaction wheels)
        #   3 - Downlink (downlink collected imagery to ground station)

        self.action_space = spaces.Discrete(4)
        print(self.action_space.n)

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
            self.simulator = leoNadirSimulator.LEONadirSimulator(.1, 1.0, self.step_duration, mass=self.mass, powerDraw = self.powerDraw)
            self.simulator_init = 1

        if self.curr_step >= self.max_length:
            self.episode_over = True

        prev_ob = self._get_state()
        self._take_action(action)

        reward = self._get_reward()
        self.reward_total += reward
        ob = self._get_state()

        #   If the wheel speeds get too large, end the episode.
        if ob[2] > 4000*mc.RPM:
            self.episode_over = True
            reward -= self.failure_penalty
            self.reward_total -= self.failure_penalty
            print("Died from wheel explosion. RPMs were norm:"+str(ob[2])+", limit is "+str(4000*mc.RPM)+", body rate was "+str(ob[1])+"action taken was "+str(action)+", env step"+str(self.curr_step))
            print("Prior state was RPM:"+str(prev_ob[2])+" . body rate was:"+str(prev_ob[1]))


        #   If we run out of power, end the episode.
        if ob[3] == 0:
            self.episode_over = True
            reward -= self.failure_penalty
            self.reward_total -= self.failure_penalty
            print("Ran out of power. Battery level at:"+str(ob[3])+", env step"+str(self.curr_step))

        #   If we overflow the buffer, end the episode.
        if ob[5] > self.simulator.storageUnit.storageCapacity:
            self.episode_over = True
            reward -= self.failure_penalty
            self.reward_total -= self.failure_penalty
            print("Data buffer overflow. Data storage level at:"+str(ob[5])+", env step"+str(self.curr_step))

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
        # If the sim is over, the agent failed, so return the zero reward (might not catch the very last step if we
        # don't mess up, but whatever for now)
        if self.sim_over and (self.curr_step != self.max_length):
            reward = 0
        # If the sim is not over and we did not fail, return reward
        else:
            reward = -self.obs[6][0]*self.reward_mult**self.curr_step

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
        # Create the simulator
        self.simulator = leoNadirSimulator.LEONadirSimulator(.1, 1.0, self.step_duration)
        # Extract initial conditions from instantiation of simulator
        self.initial_conditions = self.simulator.initial_conditions
        self.simulator_init = 1
        self.seed()

        return self.simulator.obs

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation.
        WIP: Work out which error representation to give the algo."""

        return self.simulator.obs

    def reset_init(self, initial_conditions=None):
        # If the simulate already exists, close it gracefully or you will end up w too many spice objects
        if self.simulator:
            print("Closing Spice...")
            self.simulator.close_gracefully()

        del self.simulator

        self.action_episode_memory.append([])
        self.episode_over = False
        self.curr_step = 0
        self.reward_total = 0

        # If initial conditions are passed in, use those
        if initial_conditions:
            self.initial_conditions = initial_conditions
        # Otherwise, the initial conditions should have been defined during reset()

        self.simulator = leoNadirSimulator.LEONadirSimulator(.1, 1.0, self.step_duration, self.initial_conditions)

        self.simulator_init = 1

        return self.simulator.obs
