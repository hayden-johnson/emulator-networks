import numpy as np
import torch as th

import sbi 
from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from sbi.utils.simulation_utils import simulate_for_sbi
from sbi.inference import MCABC
from sbibm.metrics import c2st

# get posterior sample from a simulator/obs using rejection sampling
def get_posterior_samples(prior, simulator, observation, n_samples=10000):
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    inference_method = MCABC(
        simulator=simulator,
        prior=prior,
        show_progress_bars=False,
    )
    output, _ = inference_method(
        x_o=observation,
        num_simulations=n_samples*100,
        quantile=.01,
        return_summary=True
    )
    return output

# 
def NLE_comparison(simulator, prior, obs, n_sims_range, n_samples=1000, n_repeats=5):
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(gaussian_simulator, prior, prior_returns_numpy)
    mcmc_samples = get_posterior_samples(prior, gaussian_simulator, obs, n_samples=n_samples)
    c2st_means, c2st_std = [],[]
    for n in tqdm(n_sims_range):
        c = []
        for _ in range(n_repeats):
            inference = NLE(prior=prior)
            theta, x = simulate_for_sbi(simulator, prior, num_simulations=n, show_progress_bar=False)
            density_estimator = inference.append_simulations(theta, x).train(show_train_summary=False)
            posterior = inference.build_posterior(density_estimator)
            nle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            c.append(c2st(nle_samples, mcmc_samples).item())
        c = np.array(c)
        c2st_means.append(c.mean())
        c2st_std.append(c.std())

    return c2st_means, c2st_std

def SNLE_comparison(simulator, prior, obs, n_sims_range, n_samples=1000, n_repeats=5):
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(gaussian_simulator, prior, prior_returns_numpy)
    mcmc_samples = get_posterior_samples(prior, gaussian_simulator, obs, n_samples=n_samples)
    c2st_means, c2st_std = [],[]
    for n in tqdm(n_sims_range):
        c = []
        n_rounds = 3
        init_pct = .5
        n_sims_init = int(n * init_pct)
        n_sims_round = int((n - n_sims_init) / n_rounds)
        for _ in range(n_repeats):
            # init training
            inference = NLE(prior=prior)
            theta, x = simulate_for_sbi(simulator, prior, num_simulations=n_sims_init, show_progress_bar=False)
            density_estimator = inference.append_simulations(theta, x).train(show_train_summary=False)
            posterior = inference.build_posterior(density_estimator)
            for _ in range(n_rounds):
                proposal = posterior.set_default_x(obs)
                theta, x = simulate_for_sbi(simulator, proposal, num_simulations=n_sims_round, show_progress_bar=False)
                density_estimator = inference.append_simulations(theta, x).train(show_train_summary=False)
                posterior = inference.build_posterior(density_estimator)
            nle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            c.append(c2st(nle_samples, mcmc_samples).item())
        c = np.array(c)
        c2st_means.append(c.mean())
        c2st_std.append(c.std())

    return c2st_means, c2st_std

def ESNLE_comparson():
    pass

def EMNETS_comparison():
    pass


if __name__ == '__main__':
    pass
