""" Packages import """
import numpy as np
import torch
import torch.nn as nn

# import jax.numpy as np
from utils import rd_argmax
from scipy.stats import norm

from math import sqrt

import jax.numpy as jnp
import jax.random as random
from sgmcmcjax.samplers import build_sgld_sampler


class ArmGaussianLinear(object):
    def __init__(self, prior_random_state=2021, reward_random_state=2022):
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following N(0, eta)
        :param arm: int
        :return: float
        """
        return np.dot(self.features[arm], self.real_theta) + self.reward_random.normal(
            0, self.eta, 1
        )

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]

    def set_context(self):
        feature_num = self.n_features // self.n_actions
        feature = self.prior_random.uniform(-self.u, self.u, (feature_num, ))
        for i in range(self.n_actions):
            self.features[i][feature_num * i: feature_num * (i+1)] = feature

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        best_arm_reward = np.max(np.dot(self.features, self.real_theta))
        return best_arm_reward * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, arm_sequence_action_sets, T):
        """
        Compute the regret of a single experiment
        :param arm_sequence_action_sets: (arm: (T,) context: (T, n_arm, n_feature))
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        arm_sequence, action_sets = arm_sequence_action_sets # (arm_sequence: (T,) action_sets: (T, n_arm, n_feature))
        print(arm_sequence)
        feature = action_sets[np.arange(T), arm_sequence] # (T, n_feature)
        expect_reward = np.dot(feature, self.real_theta) # (T, )
        best_arm_reward = np.max(np.dot(action_sets, self.real_theta), axis=-1) # (T, )
        return np.cumsum(best_arm_reward) - np.cumsum(expect_reward)


class InfiniteContextPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=1, sigma=10):
        """
        Initialization of the arms, features and theta in
        Russo, Daniel, and Benjamin Van Roy. "Learning to optimize via information-directed sampling." Operations Research 66.1 (2018): 230-252.
        :param u: float, features are drawn from a uniform Unif(-u, u)
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward N(a^T.theta, eta)
        :param sigma: float, multiplicative factor for the covariance matrix of theta which is drawn from a
        multivariate distribution N(0, sigma*I)
        """
        super(InfiniteContextPaperLinModel, self).__init__(
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.u = u
        self.eta = eta
        self.features = np.zeros((n_actions, n_features*n_actions))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features*n_actions), sigma * np.eye(n_features*n_actions)
        )
        self.alg_prior_sigma = sigma
        self.set_context()


class InfiniteContextFreqPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=1, sigma=10):
        """
        (Frequentist modification: use fixed random seed to sample arms, features and theta.)
        Initialization of the arms, features and theta in
        Russo, Daniel, and Benjamin Van Roy. "Learning to optimize via information-directed sampling." Operations Research 66.1 (2018): 230-252.
        :param u: float, features are drawn from a uniform Unif(-u, u)
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward N(a^T.theta, eta)
        :param sigma: float, multiplicative factor for the covariance matrix of theta which is drawn from a
        multivariate distribution N(0, sigma*I)
        """
        super(InfiniteContextFreqPaperLinModel, self).__init__(
            prior_random_state=0,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.u = u
        self.eta = eta
        self.features = np.zeros((n_actions, n_features*n_actions))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features*n_actions), sigma * np.eye(n_features*n_actions)
        )
        self.alg_prior_sigma = sigma
        self.set_context()


class PriorHyperLinear(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        prior_mean: float or np.ndarray = 0.,
        prior_std: float or np.ndarray = 1.,
    ):
        super().__init__()

        self.in_features, self.out_features = input_size, output_size
        # (fan-out, fan-in)
        self.weight = np.random.randn(output_size, input_size).astype(np.float32)
        self.weight = self.weight / np.linalg.norm(self.weight, axis=1, keepdims=True)

        if isinstance(prior_mean, np.ndarray):
            self.bias = prior_mean
        else:
            self.bias = np.ones(output_size, dtype=np.float32) * prior_mean

        if isinstance(prior_std, np.ndarray):
            if prior_std.ndim == 1:
                assert len(prior_std) == output_size
                prior_std = np.diag(prior_std).astype(np.float32)
            elif prior_std.ndim == 2:
                assert prior_std.shape == (output_size, output_size)
                prior_std = prior_std
            else:
                raise ValueError
        else:
            assert isinstance(prior_std, (float, int, np.float32, np.int32, np.float64, np.int64))
            prior_std = np.eye(output_size, dtype=np.float32) * prior_std

        self.weight = torch.nn.Parameter(torch.from_numpy(prior_std @ self.weight).float())
        self.bias = torch.nn.Parameter(torch.from_numpy(self.bias).float())

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight.to(x.device), self.bias.to(x.device))
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, not np.all(self.bias.cpu().detach().numpy() == 0)
        )


class HyperLinear(nn.Module):
    def __init__(
        self,
        noise_dim,
        out_features,
        prior_std: float or np.ndarray = 1.0,
        prior_mean: float or np.ndarray = 0.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0
    ):
        super().__init__()
        self.hypermodel = nn.Linear(noise_dim, out_features)
        self.priormodel = PriorHyperLinear(noise_dim, out_features, prior_mean, prior_std)
        for param in self.priormodel.parameters():
            param.requires_grad = False

        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, x, z):
        theta = self.get_theta(z)
        out = torch.mul(x, theta).sum(-1)
        return out

    def get_theta(self, z):
        theta = self.hypermodel(z)
        prior_theta = self.priormodel(z)
        theta = self.posterior_scale * theta + self.prior_scale * prior_theta
        return theta

    def regularization(self, z):
        theta = self.hypermodel(z)
        reg_loss = theta.pow(2).mean()
        return reg_loss


class ReplayBuffer:
    def __init__(self, noise_dim=2):
        self.f_list = []
        self.r_list = []
        self.z_list = []
        self.s_list = []
        self.noise_dim = noise_dim
        self.sample_num = 0

    def __len__(self):
        return self.sample_num

    def _unit_sphere_noise(self):
        noise = np.random.randn(self.noise_dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
        return noise

    def put(self, transition):
        s, f, r = transition
        z = self._unit_sphere_noise()
        self.s_list.append(s)
        self.f_list.append(f)
        self.r_list.append(r)
        self.z_list.append(z)
        self.sample_num += 1

    def get(self, shuffle=True):
        index = list(range(self.sample_num))
        if shuffle:
            np.random.shuffle(index)
        s_data, f_data, r_data, z_data \
            = np.array(self.s_list), np.array(self.f_list), np.array(self.r_list), np.array(self.z_list)
        s_data, f_data, r_data, z_data \
            = s_data[index], f_data[index], r_data[index], z_data[index]
        return s_data, f_data, r_data, z_data

    def sample(self, n):
        index = np.random.randint(low=0, high=self.sample_num, size=n)
        s_data, f_data, r_data, z_data \
            = np.array(self.s_list), np.array(self.f_list), np.array(self.r_list), np.array(self.z_list)
        s_data, f_data, r_data, z_data \
            = s_data[index], f_data[index], r_data[index], z_data[index]
        return s_data, f_data, r_data, z_data


class HyperModel:
    def __init__(
        self,
        noise_dim,
        feature_dim,
        prior_std: float or np.ndarray = 1.0,
        prior_mean: float or np.ndarray = 0.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        lr: float = 0.01,
        fg_lambda: float = 1.0,
        fg_decay: bool = True,
        batch_size: int = 32,
        optim: str = 'Adam',
        norm_coef: float = 0.01,
        target_noise_coef: float = 0.01,
        reset: bool = False,
    ):

        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.prior_std = prior_std
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.lr = lr
        self.fg_lambda = fg_lambda
        self.fg_decay = fg_decay
        self.batch_size = batch_size
        self.optim = optim
        self.norm_coef = norm_coef
        self.target_noise_coef = target_noise_coef
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.__init_model_optimizer()
        self.__init_buffer()
        self.update = getattr(self, '_update_reset') if reset else getattr(self, '_update')

    def __init_model_optimizer(self):
        self.model = HyperLinear(
            self.noise_dim, self.feature_dim,
            self.prior_std, self.prior_mean,
            self.prior_scale, self.posterior_scale
        ).to(self.device) # init hypermodel
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) # init optimizer
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9) # init optimizer
        else:
            raise NotImplementedError

    def __init_buffer(self):
        self.buffer = ReplayBuffer(noise_dim=self.noise_dim) # init replay buffer

    def put(self, transition):
        self.buffer.put(transition)

    def _update(self):
        s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        self.learn(s_batch, f_batch, r_batch, z_batch)

    def _update_reset(self):
        sample_num = len(self.buffer)
        if sample_num > self.batch_size:
            s_data, f_data, r_data, z_data = self.buffer.get()
            for i in range(0, self.batch_size, sample_num):
                s_batch, f_batch, r_batch, z_batch \
                    = s_data[i:i+self.batch_size], f_data[i:i+self.batch_size], r_data[i: i+self.batch_size], z_data[i:i+self.batch_size]
                self.learn(s_batch, f_batch, r_batch, z_batch)
            if sample_num % self.batch_size !=0:
                last_sample = sample_num % self.batch_size
                index1 = -np.arange(1, last_sample + 1).astype(np.int32)
                index2 = np.random.randint(low=0, high=sample_num, size=self.batch_size-last_sample)
                index = np.hstack([index1, index2])
                s_batch, f_batch, r_batch, z_batch = s_data[index], f_data[index], r_data[index], z_data[index]
                self.learn(s_batch, f_batch, r_batch, z_batch)
        else:
            s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
            self.learn(s_batch, f_batch, r_batch, z_batch)

    def learn(self, s_batch, f_batch, r_batch, z_batch):
        z_batch = torch.FloatTensor(z_batch).to(self.device)
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        s_batch = torch.FloatTensor(s_batch).to(self.device)

        update_noise = self.generate_noise(self.batch_size) # sample noise for update
        target_noise = torch.mul(z_batch, update_noise).sum(-1) * self.target_noise_coef # noise for target
        theta = self.model.get_theta(update_noise)

        fg_lambda = self.fg_lambda / np.sqrt(len(self.buffer)) if self.fg_decay else self.fg_lambda
        norm_coef = self.norm_coef / len(self.buffer)
        fg_term = torch.einsum('bd,bad -> ba', theta, s_batch).max(dim=-1)[0]
        predict = self.model(f_batch, update_noise)
        diff = target_noise + r_batch - predict
        loss = (diff.pow(2) - fg_lambda * fg_term).mean()
        reg_loss = self.model.regularization(update_noise) * norm_coef

        loss += reg_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample_theta(self, M):
        action_noise = self.generate_noise(M)
        with torch.no_grad():
            thetas = self.model.get_theta(action_noise).cpu().numpy()
        return thetas

    def generate_noise(self, batch_size):
        noise = torch.randn(batch_size, self.noise_dim).type(torch.float32).to(self.device)
        # noise = noise / torch.norm(noise, dim=1, keepdim=True)
        return noise

    def reset(self):
        self.__init_model_optimizer()


class InfiniteContextLinMAB:
    def __init__(self, model):
        """
        :param model: ArmGaussianLinear object
        """
        self.model = model
        self.regret, self.expect_regret, self.n_a, self.d, self.features = (
            model.regret,
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

    def set_context(self):
        self.model.set_context()
        self.features = self.model.features

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
            np.linalg.inv(s_inv + ffT / self.eta ** 2),
            np.dot(s_inv, mu) + r * f / self.eta ** 2,
        )
        sigma_ = np.linalg.inv(s_inv + ffT / self.eta ** 2)
        return r, mu_, sigma_

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            self.set_context()
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
        return reward, (arm_sequence, action_sets)

    def TS_hyper(self, T, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=2):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = model.sample_theta(1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def TS_hyper_reset(self, T, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=1):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef, reset=True
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        for t in range(T):
            self.set_context()
            # print(t)
            # print(sigma_t)
            # if np.isnan(mu_t, sigma_t).any():
            #     print(mu_t, sigma_t)
            theta_t = model.sample_theta(1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            model.reset()
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def LinUCB(self, T, lbda=10e-4, alpha=10e-1):
        """
        Implementation of Linear UCB algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param lbda: float, regression regularization parameter
        :param alpha: float, tunable parameter to control between exploration and exploitation
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
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
            arm_sequence[t], reward[t] = a_t, r_t
        return reward, arm_sequence

    def BayesUCB(self, T):
        """
        Implementation of Bayesian Upper Confidence Bounds (BayesUCB) algorithm for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            a_t = rd_argmax(
                np.dot(self.features, mu_t)
                + norm.ppf(t / (t + 1))
                * np.sqrt(
                    np.diagonal(np.dot(np.dot(self.features, sigma_t), self.features.T))
                )
            )
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

    def GPUCB(self, T):
        """
        Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and
        Experimental Design' for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
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
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

    def Tuned_GPUCB(self, T, c=0.9):
        """
        Implementation of Tuned GPUCB described in Russo & Van Roy's paper of study for Linear Bandits with
        multivariate normal prior
        :param T: int, time horizon
        :param c: float, tunable parameter. Default 0.9
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
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
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t] = r_t, a_t
        return reward, arm_sequence

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
                [
                    np.mean([theta_hat_[a]], axis=1).squeeze()
                    for a in range(self.n_a)
                ]
            )
        )
        L_hat = np.sum(
            np.array(
                [
                    p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu)
                    for a in range(self.n_a)
                ]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [
                    p_a[a] * np.dot(self.features[a], mu_a[a])
                    for a in range(self.n_a)
                ]
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
        arm = rd_argmax(-(delta ** 2) / (v + 1e-20))
        return arm, p_a

    def solveVIDS(self, thetas):
        M = thetas.shape[0]
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        theta_hat_ = [thetas[np.where(theta_hat == a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)]) / M
        mu_a = np.nan_to_num(
            np.array(
                [
                    np.mean([theta_hat_[a]], axis=1).squeeze()
                    for a in range(self.n_a)
                ]
            )
        )
        L_hat = np.sum(
            np.array(
                [
                    p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu)
                    for a in range(self.n_a)
                ]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [
                    p_a[a] * np.dot(self.features[a], mu_a[a])
                    for a in range(self.n_a)
                ]
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
        prob = np.zeros(shape=(self.n_a, self.n_a))
        psi = np.ones(shape=(self.n_a, self.n_a)) * np.inf
        for i in range(self.n_a-1):
            for j in range(i+1, self.n_a):
                if delta[j] < delta[i]:
                    D1, D2, I1, I2, flip = delta[j], delta[i], v[j], v[i], True
                else:
                    D1, D2, I1, I2, flip = delta[i], delta[j], v[i], v[j], False
                p = np.clip((D1 / (D2 - D1)) - (2 * I1 / (I2 - I1)), 0., 1.) if I1 < I2 else 0.
                psi[i][j] = ((1 - p) * D1 + p * D2)**2 / ((1 - p) * I1 + p * I2 + 1e-20)
                prob[i][j] = 1 - p if flip else p
        psi = psi.flatten()
        optim_indexes = np.nonzero(psi == psi.min())[0].tolist()
        optim_index = np.random.choice(optim_indexes)
        optim_index = [optim_index // self.n_a, optim_index % self.n_a]
        optim_prob = prob[optim_index[0], optim_index[1]]
        arm = np.random.choice(optim_index, p=[1 - optim_prob, optim_prob])
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
        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            self.set_context()
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
                a_t, p_a = self.computeVIDS(thetas)
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_hyper(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=2):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            self.set_context()
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                thetas = model.sample_theta(M)
                a_t, p_a = self.computeVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_hyper_reset(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=1):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef, reset=True
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            self.set_context()
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                thetas = model.sample_theta(M)
                a_t, p_a = self.computeVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            model.reset()
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_solution(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        mu_t, sigma_t = self.initPrior()
        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            self.set_context()
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
                a_t, p_a = self.solveVIDS(thetas)
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_solution_hyper(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=2):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            self.set_context()
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                thetas = model.sample_theta(M)
                a_t, p_a = self.solveVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def VIDS_sample_solution_hyper_reset(self, T, M=10000, noise_dim=2, fg_lambda=1.0, fg_decay=True, lr=0.01, batch_size=32, optim='Adam', update_num=1):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        norm_coef = (self.eta / self.prior_sigma)**2
        model = HyperModel(
            noise_dim, self.d, prior_std=self.prior_sigma,
            fg_lambda=fg_lambda, fg_decay=fg_decay, lr=lr, batch_size=batch_size, optim=optim,
            target_noise_coef=self.eta, norm_coef=norm_coef, reset=True
        )

        action_sets, arm_sequence, reward = np.zeros((T, self.n_a, self.d)), np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            self.set_context()
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                thetas = model.sample_theta(M)
                a_t, p_a = self.solveVIDS(thetas)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], arm_sequence[t], action_sets[t] = r_t, a_t, self.features
            model.put((self.features, f_t, r_t))
            # update hypermodel
            model.reset()
            for _ in range(update_num):
                model.update()
        return reward, (arm_sequence, action_sets)

    def SGLD_Sampler(self, X, y, n_samples, n_iters, fg_lambda):
        assert n_iters >= n_samples + 99
        # define model in JAX
        def loglikelihood(theta, x, y):
            return -(
                1 / (2 * (self.eta ** 2)) * (jnp.dot(x, theta) - y) ** 2
            ) + fg_lambda * jnp.max(jnp.dot(self.features, theta))

        def logprior(theta):
            return -0.5 * jnp.dot(theta, theta) * (1 / self.prior_sigma)

        # generate random key in jax
        key = random.PRNGKey(np.random.randint(1, 312414))
        # print("fg_lambda={}".format(fg_lambda), key)

        dt = 1e-5
        N = X.shape[0]
        # print(N)
        batch_size = int(0.1*(N+9))
        # print(batch_size)
        my_sampler = build_sgld_sampler(
            dt, loglikelihood, logprior, (X, y), batch_size, pbar=False
        )
        # run sampler
        key, subkey = random.split(key)
        thetas = my_sampler(subkey, n_iters, jnp.zeros(self.d))
        # if jnp.isnan(thetas).any():
        #     print(jnp.where(jnp.isnan(thetas)))
        sample_idx = 100 * jnp.arange(n_samples) + 10000
        # print(sample_idx)
        return thetas[sample_idx]

    def FGTS(self, T, fg_lambda=1.0):
        """
        Implementation of Feel-Good Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :param fg_lambda: float, coefficient for feel good term
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """

        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)

        for t in range(T):
            if t == 0:
                mu_t, sigma_t = self.initPrior()
                theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            else:
                X = jnp.asarray(self.features[arm_sequence[:t]])
                y = jnp.asarray(reward[:t])
                theta_t = self.SGLD_Sampler(
                    X, y, n_samples=1, n_iters=max(100 * 1 + 10000, t), fg_lambda=fg_lambda
                ).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            reward[t], arm_sequence[t] = self.reward(a_t)[0], a_t
        return reward, arm_sequence

    def FGTS01(self, T):
        """
        Implementation of Feel-Good Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior and posterios sampling via SGMCMC
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.FGTS(T, fg_lambda=10)

    def TS_SGMCMC(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior and posterios sampling via SGMCMC
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.FGTS(T, fg_lambda=0)

    def VIDS_sample_sgmcmc_fg(self, T, M=10000, fg_lambda=1):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        arm_sequence, reward = np.zeros(T, dtype=int), np.zeros(T)
        p_a = np.zeros(self.n_a)
        for t in range(T):
            # print("max posterior probability of action: {}".format(np.max(p_a)))
            if np.max(p_a) >= self.threshold:
                # print("stop")
                # Stop learning policy
                a_t = np.argmax(p_a)
            else:
                if t == 0:
                    mu_t, sigma_t = self.initPrior()
                    thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
                else:
                    X = jnp.asarray(self.features[arm_sequence[:t]])
                    y = jnp.asarray(reward[:t])
                    thetas = self.SGLD_Sampler(X, y, M, max(100 * M + 10000, t), fg_lambda)
                a_t, p_a = self.computeVIDS(thetas)
            reward[t], arm_sequence[t] = self.reward(a_t)[0], a_t
        return reward, arm_sequence

    def VIDS_sample_sgmcmc(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.VIDS_sample_sgmcmc_fg(T, M, 0)

    def VIDS_sample_sgmcmc_fg01(self, T, M=10000):
        """
        Implementation of V-IDS with approximation of integrals using SGMCMC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        """
        return self.VIDS_sample_sgmcmc_fg(T, M, 0.1)
