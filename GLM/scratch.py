



_________________


#TODO: model averaging, wAIC, LOOCV
#TODO: ARD per coefficient to select N_bases
#TODO: lambda optimization (

# Bayes MCMC
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
import numpy as np
import patsy
import jax
import jax.numpy as jnp





#### FOR Posterior predictive checks:

# Get posterior samples
samples = mcmc.get_samples()

# Set up predictive distribution
predictive = Predictive(model_with_wiggliness_prior_optimized, posterior_samples=samples)

# Generate posterior predictive samples
y_pred = predictive(jax.random.PRNGKey(1), basis_x_list=basis_x_list, S_list=S_list)
# TODO: Compare predicted y_pred['y'] to your actual y values









______________________________________

