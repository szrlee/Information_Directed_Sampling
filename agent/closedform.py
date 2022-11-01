""" Packages import """
import numpy as np
from utils import rd_argmax, haar_matrix, sphere_matrix, multi_haar_matrix
from scipy.stats import norm
from scipy.linalg import sqrtm

# from scipy.stats import multivariate_normal


class LinMAB:
    def __init__(self, model):
        """
        :param model: ArmGaussianLinear object
        """
        self.model = model
        self.expect_regret, self.n_a, self.d, self.features = (
            model.expect_regret,
            model.n_actions,
            model.n_features,
            model.features,
        )
        self.reward, self.eta = model.reward, model.eta
        self.prior_sigma = model.alg_prior_sigma
        # self.flag = False
        # self.optimal_arm = None
        self.threshold = 0.999
        self.store_IDS = False
        # Only for Haar matrix
        self.counter = 0

    def initPrior(self):
        a0 = 0
        s0 = self.prior_sigma
        mu_0 = a0 * np.ones(self.d)
        sigma_0 = s0 * np.eye(
            self.d
        )  # to adapt according to the true distribution of theta
        return mu_0, sigma_0

    def updatePosterior(self, a, mu, sigma):
        """
        Update posterior mean and covariance matrix
        :param arm: int, arm chose
        :param mu: np.array, posterior mean vector
        :param sigma: np.array, posterior covariance matrix
        :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
        """
        f, r = self.features[a], self.reward(a)[0]
        s_inv = np.linalg.inv(sigma)
        ffT = np.outer(f, f)
        mu_ = np.dot(
            np.linalg.inv(s_inv + ffT / self.eta**2),
            np.dot(s_inv, mu) + r * f / self.eta**2,
        )
        sigma_ = np.linalg.inv(s_inv + ffT / self.eta**2)
        return r, mu_, sigma_

    def rankone_update(self, f, Sigma):
        ffT = np.outer(f, f)
        return Sigma - (Sigma @ ffT @ Sigma) / (self.eta**2 + f @ Sigma @ f)

    def updatePosterior_(self, a, sigma, p):
        """
        Update posterior mean and covariance matrix incrementally
        without matrix inversion
        :param arm: int, arm chose
        :param sigma: np.array, posterior covariance matrix
        :param p: np.array, sigma_inv mu
        :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
        """
        f, r = self.features[a], self.reward(a)[0]
        sigma_ = self.rankone_update(f, sigma)
        p_ = p + ((r * f) / self.eta**2)
        mu_ = sigma_ @ p_
        return r, mu_, sigma_, p_

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            try:
                theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            except:
                try:
                    # print(np.isnan(sigma_t).any(), np.isinf(sigma_t).any())
                    z = np.random.normal(0, 1, self.d)
                    theta_t = sqrtm(sigma_t) @ z + mu_t
                except:
                    print(np.isnan(sigma_t).any(), np.isinf(sigma_t).any())

            a_t = rd_argmax(np.dot(self.features, theta_t))
            r_t, mu_t, sigma_t, p_t = self.updatePosterior_(a_t, sigma_t, p_t)
            # r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)

            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def ES(self, T, M=10):
        """
        Ensemble sampling
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, Sigma_t = self.initPrior()
        # A_t = np.zeros((self.d, M))
        # P_t = np.zeros((self.d, M))
        B = np.random.normal(0, 1, (self.d, M))
        # print(B.shape, np.linalg.norm(B, axis=1))
        A_t = mu_t.reshape((self.d, 1)) + sqrtm(Sigma_t) @ B
        P_t = np.linalg.inv(Sigma_t) @ A_t
        for t in range(T):
            i = np.random.choice(M, 1)[0]
            theta_t = A_t[:, [i]]
            a_t = rd_argmax(np.dot(self.features, theta_t))
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            Sigma_t = self.rankone_update(f_t, Sigma_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)
            # Update M models with algorithmic random perturbation
            P_t += np.outer(
                f_t, (r_t + np.random.normal(0, self.eta, M)) / self.eta**2
            )
            A_t = Sigma_t @ P_t

        return reward, expected_regret

    def rand_vec_gen(self, M, N=1, haar=False):
        """
        Random vector generation
        """
        assert N > 0
        if not haar:
            return sphere_matrix(N, M)
        # generate vectors from Haar random matrix
        if N == 1:
            if self.counter == 0:
                self.V = haar_matrix(M)
            v = self.V[:, self.counter]
            self.counter = (self.counter + 1) % M
            return v
        else:
            return multi_haar_matrix(N, M)

    def IS(self, T, M=10, haar=False):
        """
        Index Sampling
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, Sigma_t = self.initPrior()
        # Sample matrix B with size d x M
        B = self.rand_vec_gen(M, N=self.d, haar=haar)
        # print(B.shape, np.linalg.norm(B, axis=1))
        A_t = sqrtm(Sigma_t) @ B
        S_inv = np.linalg.inv(Sigma_t)
        P_t = S_inv @ A_t
        p_t = S_inv @ mu_t

        for t in range(T):
            # index sampling
            z = np.random.normal(0, 1, M)
            theta_t = A_t @ z + mu_t
            # action selection
            a_t = rd_argmax(np.dot(self.features, theta_t))
            # selected feature and reward feedback
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            # incremental update on Sigma and mu
            # Sigma_t = self.rankone_update(f_t, Sigma_t)
            r_t, mu_t, Sigma_t, p_t = self.updatePosterior_(a_t, Sigma_t, p_t)
            # compute regret
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)
            # Update A
            b_t = self.rand_vec_gen(M, haar=haar)
            P_t += np.outer(f_t, b_t) / self.eta
            A_t = Sigma_t @ P_t

        return reward, expected_regret

    def LinUCB(self, T, lbda=10e-4, alpha=10e-1):
        """
        Implementation of Linear UCB algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param lbda: float, regression regularization parameter
        :param alpha: float, tunable parameter to control between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        a_t, A_t, b_t = (
            np.random.randint(0, self.n_a - 1, 1)[0],
            lbda * np.eye(self.d),
            np.zeros(self.d),
        )
        r_t = self.reward(a_t)
        for t in range(T):
            A_t += np.outer(self.features[a_t, :], self.features[a_t, :])
            b_t += r_t * self.features[a_t, :]
            inv_A = np.linalg.inv(A_t)
            theta_t = np.dot(inv_A, b_t)
            beta_t = alpha * np.sqrt(
                np.diagonal(np.dot(np.dot(self.features, inv_A), self.features.T))
            )
            a_t = rd_argmax(np.dot(self.features, theta_t) + beta_t)
            r_t = self.reward(a_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def BayesUCB(self, T):
        """
        Implementation of Bayesian Upper Confidence Bounds (BayesUCB) algorithm for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + norm.ppf(t / (t + 1))
                * np.sqrt(
                    np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))
                )
            )
            r_t, mu_t, sigma_t, p_t = self.updatePosterior_(a_t, sigma_t, p_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def GPUCB(self, T):
        """
        Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and
        Experimental Design' for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            beta_t = 2 * np.log(self.n_a * ((t + 1) * np.pi) ** 2 / 6 / 0.1)
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + np.sqrt(
                    beta_t
                    * np.diagonal(
                        np.dot(np.dot(self.features, sigma_t), self.features.T)
                    )
                )
            )
            r_t, mu_t, sigma_t, p_t = self.updatePosterior_(a_t, sigma_t, p_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def Tuned_GPUCB(self, T, c=0.9):
        """
        Implementation of Tuned GPUCB described in Russo & Van Roy's paper of study for Linear Bandits with
        multivariate normal prior
        :param T: int, time horizon
        :param c: float, tunable parameter. Default 0.9
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        for t in range(T):
            beta_t = c * np.log(t + 1)
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + np.sqrt(
                    beta_t
                    * np.diagonal(
                        np.dot(np.dot(self.features, sigma_t), self.features.T)
                    )
                )
            )
            r_t, mu_t, sigma_t, p_t = self.updatePosterior_(a_t, sigma_t, p_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def computeVIDS(self, thetas):
        """
        Implementation of linearSampleVIR (algorithm 6 in Russo & Van Roy, p. 244) applied for Linear  Bandits with
        multivariate normal prior. Here integrals are approximated in sampling thetas according to their respective
        posterior distributions.
        :param thetas: np.array, posterior samples
        :return: int, np.array, arm chose and p*
        """
        # print(thetas.shape)
        M = thetas.shape[0]
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        # print("theta_hat shape: {}".format(theta_hat.shape))
        theta_hat_ = [thetas[np.where(theta_hat == a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)]) / M
        # if np.max(p_a) >= self.threshold:
        #     # Stop learning policy
        #     self.optimal_arm = np.argmax(p_a)
        #     arm = self.optimal_arm
        # else:
        # print("theta_hat_[0]: {}, theta_hat_[0] length: {}".format(theta_hat_[0], len(theta_hat_[0])))
        mu_a = np.nan_to_num(
            np.array(
                [np.mean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]
            )
        )
        L_hat = np.sum(
            np.array(
                [p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu) for a in range(self.n_a)]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [p_a[a] * np.dot(self.features[a], mu_a[a]) for a in range(self.n_a)]
            ),
            axis=0,
        )
        v = np.array(
            [
                np.dot(np.dot(self.features[a], L_hat), self.features[a].T)
                for a in range(self.n_a)
            ]
        )
        delta = np.array(
            [rho_star - np.dot(self.features[a], mu) for a in range(self.n_a)]
        )
        arm = rd_argmax(-(delta**2) / (v + 1e-20))
        return arm, p_a

    def VIDS_sample(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        mu_t, sigma_t = self.initPrior()
        p_t = np.linalg.inv(sigma_t) @ mu_t
        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
            a_t, p_a = self.computeVIDS(thetas)
            r_t, mu_t, sigma_t, p_t = self.updatePosterior_(a_t, sigma_t, p_t)
            # r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret
