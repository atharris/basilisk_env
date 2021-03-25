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
        self.max_length = int(270) # Specify the maximum number of minutes
        self.max_steps = 45

        #   Tell the environment that it doesn't have a sim attribute...
        self.simulator_init = 0
        self.simulator = None
        self.reward_total = 0

        # Set initial conditions to none (gets assigned in reset)
        self.initial_conditions = None

        # Set the dynRate for the env, which is passed into the simulator
        self.dynRate = 1.0
        self.fswRate = 1.0

        #   Set up options, constants for this environment
        self.step_duration = 6*60. # seconds, tune as desired
        self.reward_mult = 1.0
        self.failure_penalty = 1000
        low = -1e16
        high = 1e16
        self.observation_space = spaces.Box(low, high,shape=(23,1))
        self.obs = np.zeros([23,1])
        self.obs_full = np.zeros([23,1])

        self.action_space = spaces.Discrete(4)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []
        self.curr_step = 0
        self.episode_over = False
        self.failure = False

    def _seed(self):
        np.random.seed()
        return

    def step(self, action, return_obs=True):
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
            self.simulator = leoNadirSimulator.LEONadirSimulator(self.dynRate, self.fswRate, self.step_duration)
            self.simulator_init = 1

        # If the simTime in minutes is greater than the planning interval in minutes, end the sim
        if (self.simulator.simTime/60) >= self.max_length:
            print("End of simulation reached", self.simulator.simTime/60)
            self.episode_over = True

        prev_ob = self.obs_full
        self._take_action(action, return_obs)

        # If we want to return observations, do the following
        if return_obs:
            reward = 0
            ob = self._get_state()

            #   If the wheel speeds get too large, end the episode.
            if any(speeds > 6000*mc.RPM for speeds in self.obs_full[8:10]):
                self.episode_over = True
                self.failure = True
                reward -= self.failure_penalty
                self.reward_total -= self.failure_penalty
                print("Died from wheel explosion. RPMs were norm:"+self.obs_full[8:10]+", limit is "+str(6000*mc.RPM)+", body rate was "+str(self.obs_full[7])+"action taken was "+str(action)+", env step"+str(self.curr_step))
                print("Prior state was RPM:"+prev_ob[8:10]+" . body rate was:"+str(prev_ob[7]))

            #   If we run out of power, end the episode.
            elif self.obs_full[11] == 0:
                self.failure = True
                self.episode_over = True
                reward -= self.failure_penalty
                self.reward_total -= self.failure_penalty
                print("Ran out of power. Battery level at: "+str(self.obs_full[11])+", env step "+str(self.curr_step)+" action taken was "+str(action))

            #   If we overflow the buffer, end the episode.
            elif self.obs_full[13] >= self.simulator.storageUnit.storageCapacity:
                self.failure = True
                self.episode_over = True
                reward -= self.failure_penalty
                self.reward_total -= self.failure_penalty
                print("Data buffer overflow. Data storage level at:"+str(self.obs_full[13])+", env step"+str(self.curr_step))

            elif self.sim_over:
                self.episode_over = True
                print("Orbit decayed - no penalty, but this one is over.")

            else:
                self.failure = False

            if self.episode_over:
                info = {'episode':{
                    'r': self.reward_total,
                    'l': self.curr_step},
                    'obs': ob
                }
            else:
                info={
                    'obs': ob
                }
            reward = self._get_reward()
            self.reward_total += reward

        # Otherwise, return nothing
        else:
            ob = []
            reward = 0
            info = {}

        self.curr_step += 1
        return ob, reward, self.episode_over, info

    def _take_action(self, action, return_obs=True):
        '''
        Interfaces with the simulator to
        :param action:
        :return:
        '''

        self.action_episode_memory[self.curr_episode].append(action)
        self.obs, self.sim_over, self.obs_full = self.simulator.run_sim(action, return_obs)

    def _get_reward(self):
        """
        Reward is based on time spent with the inertial attitude pointed towards the ground within a given tolerance.

        """
        if self.failure:
            reward = -self.failure_penalty
        elif self.episode_over:
            reward = (-self.obs_full[14][0])*(self.reward_mult**self.curr_step)+1
        else:
            reward = -self.obs_full[14][0]*(self.reward_mult**self.curr_step)

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
        self.failure = False
        self.curr_step = 0
        self.reward_total = 0

        # Create the simulator
        self.simulator = leoNadirSimulator.LEONadirSimulator(self.dynRate, self.fswRate, self.step_duration)

        # Extract initial conditions from instantiation of simulator
        self.initial_conditions = self.simulator.initial_conditions
        self.simulator.max_steps = self.max_steps
        self.simulator.max_length = self.max_length
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

        del self.simulator

        self.action_episode_memory.append([])
        self.episode_over = False
        self.curr_step = 0
        self.reward_total = 0
        self.failure = False

        # If initial conditions are passed in, use those
        if initial_conditions:
            self.initial_conditions = initial_conditions

        self.simulator = leoNadirSimulator.LEONadirSimulator(self.dynRate, self.fswRate, self.step_duration, self.initial_conditions)
        self.simulator.max_steps = self.max_steps
        self.simulator.max_length = self.max_length

        self.simulator_init = 1

        return self.simulator.obs
