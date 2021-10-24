# %% [markdown]
## Example for stochastic gradient MCMC
#### To add a new cell, type '# %%'
#### To add a new markdown cell, type '# %% [markdown]'
# %%
import jax.numpy as jnp
from jax import random
from sgmcmcjax.samplers import build_sgld_sampler


# define model in JAX
def loglikelihood(theta, x, y):
    return -((jnp.dot(x, theta) - y) ** 2)

def logprior(theta):
    return -0.5 * jnp.dot(theta, theta) * 100

# generate dataset
N, D = 10_000, 100
key = random.PRNGKey(0)
X_data = random.normal(key, shape=(N, D))
y_data = random.normal(key, shape=(N, ))

# build sampler
batch_size = int(0.1*N)
dt = 1e-5
my_sampler = build_sgld_sampler(dt, loglikelihood, logprior, (X_data, y_data), batch_size)

# run sampler
Nsamples = 10_000
samples = my_sampler(key, Nsamples, jnp.zeros(D))


