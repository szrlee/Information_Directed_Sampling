""" Packages import """
from agent.MAB import *
import numpy as np
from utils import rd_argmax
import random


class oldMAB(GenericMAB):
    """
    Old MAB class for arms that defines general methods
    """

    def __init__(self, env, p):
        super().__init__(envs=[env] * len(p), p=p)

    def RandomPolicy(self, T):
        """
        Implementation of a random policy consisting in randomly choosing one of the available arms. Only useful
        for checking that the behavior of the different policies is normal
        :param T:  int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence, expected_regret = self.init_lists(T)
        for t in range(T):
            arm = random.randint(0, self.nb_arms - 1)
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, expected_regret)

        return reward, expected_regret

    def ExploreCommit(self, T, m):
        """
        Implementation of Explore-then-Commit algorithm
        :param T: int, time horizon
        :param m: int, number of rounds before choosing the best action
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence, expected_regret = self.init_lists(T)
        for t in range(m * self.nb_arms):
            arm = t % self.nb_arms
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, expected_regret)
        arm = np.argmax(Sa / Na)
        for t in range(m * self.nb_arms + 1, T):
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, expected_regret)

        return reward, expected_regret

    def UCB1(self, T, rho):
        """
        Implementation of UCB1 algorithm
        :param T: int, time horizon
        :param rho: float, parameter for balancing between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence, expected_regret = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                arm = rd_argmax(Sa / Na + rho * np.sqrt(np.log(t + 1) / 2 / Na))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, expected_regret)

        return reward, expected_regret

    def UCB_Tuned(self, T):
        """
        Implementation of UCB-tuned algorithm
        :param T: int, time horizon
        :param rho: float, parameter for balancing between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence, expected_regret = self.init_lists(T)
        S, m = np.zeros(self.nb_arms), np.zeros(self.nb_arms)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                for arm in range(self.nb_arms):
                    S[arm] = (
                        sum([r**2 for r in reward[np.where(arm_sequence == arm)]])
                        / Na[arm]
                        - (Sa[arm] / Na[arm]) ** 2
                    )
                    m[arm] = min(0.25, S[arm] + np.sqrt(2 * np.log(t + 1) / Na[arm]))
                arm = rd_argmax(Sa / Na + np.sqrt(np.log(t + 1) / Na * m))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, expected_regret)

        return reward, expected_regret

    def MOSS(self, T, rho):
        """
        Implementation of Minimax Optimal Strategy in the Stochastic case (MOSS).
        :param T: int, time horizon
        :param rho: float, parameter for balancing between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        Sa, Na, reward, arm_sequence, expected_regret = self.init_lists(T)
        for t in range(T):
            if t < self.nb_arms:
                arm = t
            else:
                root_term = np.array(
                    list(map(lambda x: max(x, 1), T / (self.nb_arms * Na)))
                )
                arm = rd_argmax(Sa / Na + rho * np.sqrt(4 / Na * np.log(root_term)))
            self.update_lists(t, arm, Sa, Na, reward, arm_sequence, expected_regret)

        return reward, expected_regret
