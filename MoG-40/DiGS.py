import sys
sys.path.append("../")

import torch
import numpy as np
from tqdm import tqdm

# samplers/DiffusiveGibbs.py
from samplers.DiffusiveGibbs import langevin_diffusive_gibbs_sampler_step
# utils.py
from utils import GMM, quadratic_function, MC_estimate_true_expectation, relative_mae, plot_contours
# ../mmd.py
from mmd import MMD_loss

device = torch.device("cpu")
torch.set_default_tensor_type('torch.FloatTensor')

dim = 2
n_mixes = 40
loc_scaling = 40.0 # scale of the problem (changes how far apart the modes of each Gaussian component will be)
log_var_scaling = 1.0 # variance of each Gaussian

torch.manual_seed(0) # seed of 0 for GMM problem
target = GMM(dim=dim, n_mixex=n_mixes, 
             loc_scaling=loc_scaling, log_var_scaling=log_var_scaling,
             device=device)

energy = lambda x: -target.log_prob(x)

plotting_bounds = (-loc_scaling * 1.4, loc_scaling * 1.4)

n_samples = int(1e4)
true_samples = target.sample([n_samples])
plot_contours(target.log_prob, samples=true_samples.cpu().numpy(),
              bounds=plotting_bounds, n_contour_levels=50, grid_width_n_points=200,
              device=device, plt_show=False)

true_expectation = MC_estimate_true_expectation(true_samples.cpu(), quadratic_function)
print("True expectation: ", true_expectation.cpu().numpy())