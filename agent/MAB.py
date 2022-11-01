""" Packages import """
import numpy as np
from env import arms
from tqdm import tqdm
import inspect

mapping = {
    "B": arms.ArmBernoulli,
    "beta": arms.ArmBeta,
    "F": arms.ArmFinite,
    "G": arms.ArmGaussian,
}


class GenericMAB:
    """
    Generic class for arms that defines general methods
    """

    def __init__(self, methods, p):
        """
        Initialization of the arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        """
        self.MAB = self.generate_arms(methods, p)
        self.nb_arms = len(self.MAB)
        self.means = [el.mean for el in self.MAB]
        self.mu_max = np.max(self.means)
        self.IDS_results = {"arms": [], "policy": [], "delta": [], "g": [], "IR": []}
        self.store_IDS = False

    @staticmethod
    def generate_arms(methods, p):
        """
        Method for generating different arms
        :param methods: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, m in enumerate(methods):
            args = [p[i]] + [[np.random.randint(1, 312414)]]
            args = sum(args, []) if type(p[i]) == list else args
            try:
                alg = mapping[m]
                arms_list.append(alg(*args))
            except Exception:
                raise NotImplementedError
        return arms_list

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        return self.mu_max * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, reward):
        """
        Compute the regret of a single step
        """
        return self.mu_max - reward

    def MC_regret(self, method, N, T, param_dic):
        """
        Implementation of Monte Carlo method to approximate the expectation of the regret
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent Monte Carlo simulation
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods, can be the value of rho for UCB model or an int
        corresponding to the number of rounds of exploration for the ExploreCommit method
        """
        mc_regret = np.zeros(T)
        try:
            alg = self.__getattribute__(method)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T] + [param_dic[method][i] for i in args]
            for _ in tqdm(range(N), desc="Computing " + str(N) + " simulations"):
                mc_regret += self.regret(alg(*args)[0], T)
        except Exception:
            raise NotImplementedError
        return mc_regret / N

    def init_lists(self, T):
        """
        Initialization of quantities of interest used for all methods
        :param T: int, time horizon
        :return: - Sa: np.array, cumulative reward of arm a
                 - Na: np.array, number of times a has been pulled
                 - reward: np.array, rewards
                 - arm_sequence: np.array, arm chose at each step
        """
        Sa, Na, reward, arm_sequence, expected_regret = (
            np.zeros(self.nb_arms),
            np.zeros(self.nb_arms),
            np.zeros(T),
            np.zeros(T),
            np.zeros(T),
        )
        return Sa, Na, reward, arm_sequence, expected_regret

    def update_lists(self, t, arm, Sa, Na, reward, arm_sequence, expected_regret):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        Na[arm], arm_sequence[t], new_reward = Na[arm] + 1, arm, self.MAB[arm].sample()
        reward[t], Sa[arm] = new_reward, Sa[arm] + new_reward
        expected_regret[t] = self.mu_max - new_reward
