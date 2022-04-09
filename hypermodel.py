import numpy as np
import torch
import torch.nn as nn

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

