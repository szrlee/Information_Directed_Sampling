""" Packages import """
from agent.MAB import *
import numpy as np
from utils import rd_argmax
import random


class oldMAB(GenericMAB):
    """
    Old MAB class for arms that defines general methods
    """

    def __init__(self, methods, p):
        super().__init__(methods=methods, p=p)

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

    def IDSAction(self, delta, g):
        """
        Implementation of IDSAction algorithm as defined in Russo & Van Roy, p. 242
        :param delta: np.array, instantaneous regrets
        :param g: np.array, information gains
        :return: int, arm to pull
        """
        Q = np.zeros((self.nb_arms, self.nb_arms))
        IR = np.ones((self.nb_arms, self.nb_arms)) * np.inf
        q = np.linspace(0, 1, 1000)
        for a in range(self.nb_arms - 1):
            for ap in range(a + 1, self.nb_arms):
                if g[a] < 1e-6 or g[ap] < 1e-6:
                    return rd_argmax(-g)
                da, dap, ga, gap = delta[a], delta[ap], g[a], g[ap]
                qaap = q[
                    rd_argmax(
                        -((q * da + (1 - q) * dap) ** 2) / (q * ga + (1 - q) * gap)
                    )
                ]
                IR[a, ap] = (qaap * (da - dap) + dap) ** 2 / (qaap * (ga - gap) + gap)
                Q[a, ap] = qaap
        amin = rd_argmax(-IR.reshape(self.nb_arms * self.nb_arms))
        a, ap = amin // self.nb_arms, amin % self.nb_arms
        b = np.random.binomial(1, Q[a, ap])
        arm = int(b * a + (1 - b) * ap)
        if self.store_IDS:
            self.IDS_results["arms"].append(arm)
            policy = np.zeros(self.nb_arms)
            policy[a], policy[ap] = Q[a, ap], (1 - Q[a, ap])
            self.IDS_results["policy"].append(policy)
            self.IDS_results["delta"].append(delta)
            self.IDS_results["g"].append(g)
            self.IDS_results["IR"].append(
                np.inner(delta**2, policy) / np.inner(g, policy)
            )
        return arm
