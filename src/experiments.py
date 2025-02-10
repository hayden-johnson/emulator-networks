import warnings
import random 
import numpy as np
import torch as th 
from tqdm import trange

from sbi.inference import NLE
from sbi.utils import BoxUniform
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
from sbi.utils.simulation_utils import simulate_for_sbi
from sbi.inference import MCABC
from sbibm.metrics import c2st
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator
)

from emnets.simulator import gaussian_simulator
from multiprocessing import Process

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

def run_nle_comparison(simulator, prior, obs, n_sims_init, n_aquisitions, n_samples=1000, n_repeats=3, save=True):
    # validate prior and simulator for sbi 
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(gaussian_simulator, prior, prior_returns_numpy)
    # use mcmc for ground truth posterior samples 
    mcmc_samples = get_posterior_samples(prior, simulator, obs, n_samples=n_samples)
    # simulate full dataset
    theta, x = simulate_for_sbi(simulator, prior, num_simulations=n_sims_init+n_aquisitions, show_progress_bar=False)    
    C2ST = np.ones((n_repeats, n_aquisitions))*-1
    for i in range(n_repeats):
        for j in range(n_aquisitions):
            # reset & retrain NLE
            inference = NLE(prior=prior)
            density_estimator = inference.append_simulations(theta[:n_sims_init+j], x[:n_sims_init+j]).train(show_train_summary=False)
            # generate posterior samples
            posterior = inference.build_posterior(density_estimator)
            nle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            # compare MCMC with NLE w/ C2ST
            C2ST[i][j] = c2st(nle_samples, mcmc_samples).item()

    if save:
        np.savetxt("nle.txt", C2ST, delimiter=",") 
    # returns (n_repeat x n_aquisitions) matrix
    return C2ST
    
def run_snle_comparison(simulator, prior, obs, n_sims_init, n_aquisitions, n_samples=1000, n_repeats=3, save=True):
    # validate prior and simulator for sbi 
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(gaussian_simulator, prior, prior_returns_numpy)
    # use mcmc for ground truth posterior samples 
    mcmc_samples = get_posterior_samples(prior, gaussian_simulator, obs, n_samples=n_samples)
    # simulate full dataset
    theta_init, x_init = simulate_for_sbi(simulator, prior, num_simulations=n_sims_init, show_progress_bar=False)    
    C2ST = np.ones((n_repeats, n_aquisitions))*-1
    for i in range(n_repeats):
        ## initial training loop 
        inference = NLE(prior=prior)
        density_estimator = inference.append_simulations(theta_init, x_init).train(show_train_summary=False)
        posterior = inference.build_posterior(density_estimator)
        ## this line is throwing an error for some reason
        #nle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
        #print((n_samples,), type(n_samples))
        snle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
        C2ST[i][0] = c2st(snle_samples, mcmc_samples).item()
        for j in range(1, n_aquisitions):
            # set proposal distribution 
            proposal = posterior.set_default_x(obs)
            # simulate new data from proposal rather than prior
            theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1, show_progress_bar=False)
            density_estimator = inference.append_simulations(theta, x).train(show_train_summary=False)
            posterior = inference.build_posterior(density_estimator)
            snle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            # compare MCMC with SNLE w/ C2ST
            C2ST[i][j] = c2st(snle_samples, mcmc_samples).item()

    if save:
        np.savetxt("snle.txt", C2ST, delimiter=",") 

    # returns (n_repeat x n_aquisitions) matrix
    return C2ST

def run_esnle_comparison(simulator, prior, obs, n_sims_init, n_aquisitions, n_samples=1000, n_repeats=3, n_emulators=3, save=True):
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(gaussian_simulator, prior, prior_returns_numpy)
    mcmc_samples = get_posterior_samples(prior, gaussian_simulator, obs, n_samples=n_samples)
    theta_init, x_init = simulate_for_sbi(simulator, prior, num_simulations=n_sims_init, show_progress_bar=False)    
    C2ST = np.ones((n_repeats, n_aquisitions))*-1
    for i in range(n_repeats):
        emulators = [NLE(prior=prior, show_progress_bars=False) for _ in range(n_emulators)]
        ensemble = []
        for inference in emulators:
            density_estimator = inference.append_simulations(theta_init, x_init).train()
            posterior = inference.build_posterior(density_estimator)
            ensemble.append(posterior)
        ensemble_posterior = EnsemblePosterior(ensemble)
        esnle_samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
        C2ST[i][0] = c2st(esnle_samples, mcmc_samples).item()
        for j in range(1, n_aquisitions):
            proposal = ensemble_posterior.set_default_x(obs)
            theta_prop = proposal.sample((100,))
            likelihood_matrix = np.array([e.log_prob(theta_prop, obs).numpy() for e in ensemble])
            theta_star = theta_prop[np.argmax(likelihood_matrix.var(axis=0))]
            ## should we use processed simulator here?
            x = gaussian_simulator(theta_star)
            # cast as tensors
            x = th.Tensor(x).unsqueeze(-1)
            theta_star = th.Tensor(theta_star).unsqueeze(-1)
            for inference in emulators:
                density_estimator = inference.append_simulations(theta_star, x).train()
                posterior = inference.build_posterior(density_estimator)
                ensemble.append(posterior)
            ensemble_posterior = EnsemblePosterior(ensemble)
            esnle_samples = ensemble_posterior.sample((n_samples,), x=obs, show_progress_bars=False)
            C2ST[i][j] = c2st(esnle_samples, mcmc_samples).item()

    if save:
        np.savetxt("esnle.txt", C2ST, delimiter=",") 

    return C2ST


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # set experiments parameters
    obs = th.tensor([3])
    n_sims_init = 25
    n_aquisitions = 50
    n_repeats = 3

    # set prior
    prior = BoxUniform(low=-8 * th.ones(1), high=8 * th.ones(1))
    args = (gaussian_simulator, prior, obs, n_sims_init, n_aquisitions)

    nle = Process(target=run_nle_comparison, args=args)
    snle = Process(target=run_snle_comparison, args=args)
    esnle = Process(target=run_esnle_comparison, args=args)

    print("starting processes...")
    nle.start()
    snle.start()
    esnle.start()

    nle.join()
    snle.join()
    esnle.join()

    print("all processes have completed!")

   