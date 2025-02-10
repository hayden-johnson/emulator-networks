import numpy as np
import torch as th
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.uniform import Uniform
from Emulator import EmulatorNet
from train import train_network 
from simulator import gaussian_simulator


def SNLE_ensemble(x_obs,                # emperical observation(s) of X
                  prior,                # prior distribution over theta 
                  n_rounds,             # number of AL rounds
                  n_emulators,          # size of your ensemble
                  train_size,           # number of training simulations
                  valid_size,           # number of validation simulations
                  train_batch_size,     # batch size for initial training
                  train_n_epochs,       # number of epochs for iniital training
                  al_n_samples,         # number of simulations for each AL round
                  al_batch_size,        # batch size for AL training
                  al_n_epochs,          # number of epochs for AL
                  lr                    # learning rate (might want to split this for train, AL)
                  ):
    ''' Sequential neural likelihood estimation with deep ensembles '''
    # sample initial dataset
    theta_train = th.Tensor(prior.sample(sample_shape=(train_size,)))
    x_train = th.Tensor(np.array([gaussian_simulator(t) for t in theta_train]).squeeze())
    trainset = TensorDataset(theta_train.unsqueeze(-1), x_train)
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

    # sample validation set 
    theta_valid = th.Tensor(prior.sample(sample_shape=(valid_size,)))
    x_valid = th.Tensor(np.array([gaussian_simulator(t) for t in theta_valid]).squeeze())
    validset = (theta_valid, x_valid)

   ## train initial ensemble
    print('training initial ensemble...')
    likelihood_matrix = []
    ensemble, optimizers = [],[]
    init_train_loss = []
    for _ in range(n_emulators):
        emulator = EmulatorNet()
        optimizer = th.optim.Adam(emulator.parameters(), lr=lr)
        train_loss, valid_loss = train_network(emulator, optimizer, trainloader, validset, epochs=train_n_epochs, verbose=False)
        ensemble.append(emulator)
        optimizers.append(optimizer)
        # compute the likelihood of x_obs, for each theta in theta_valid
        with th.no_grad():
            probs = emulator(theta_valid.unsqueeze(-1)).log_prob(x_obs)
        likelihood_matrix.append(probs.detach().numpy())
        init_train_loss.append((train_loss, valid_loss))
    # we now have a [emulator X theta matrix] of likelihoods
    likelihood_matrix = np.array(likelihood_matrix)
    print('initial training complete!\n')
 
    ## sequential aquisition of new data 
    print('running active learning loop...')
    al_likelihoods = [likelihood_matrix]
    al_theta, al_train_loss = [], []
    for _ in trange(n_rounds):
        al_train_loss_round = []
        # choose theta from valid set with most variance across emulators 
        index = np.argmax(likelihood_matrix.var(axis=0))
        theta_new = theta_valid[index].repeat(al_n_samples)
        x_new = th.Tensor(np.array([gaussian_simulator(theta_new, n=al_n_samples)]).squeeze())
        trainset = TensorDataset(theta_new.unsqueeze(-1), x_new)
        trainloader = DataLoader(trainset, batch_size=al_batch_size, shuffle=True)
        #trainloader = create_dataloader(theta_new, x_new, batch_size=al_batch_size)
        al_theta.append(theta_new.numpy())
        # train emulators on the new data samples
        likelihood_matrix = []
        for emulator, optimizer in zip(ensemble, optimizers):
            # train only on the new samples
            train_loss, valid_loss = train_network(emulator, optimizer, trainloader, validset, epochs=al_n_epochs, verbose=False, progress_bar=False)
            probs = emulator(theta_valid.unsqueeze(-1)).log_prob(x_obs)
            likelihood_matrix.append(probs.detach().numpy())
            al_train_loss_round.append((train_loss, valid_loss))
        
        likelihood_matrix = np.array(likelihood_matrix)
        al_likelihoods.append(likelihood_matrix) 
        al_train_loss.append(al_train_loss_round)

    al_likelihoods = np.array(al_likelihoods)
    al_theta = np.array(al_theta)
    return ensemble, init_train_loss, al_train_loss, al_likelihoods

if __name__ == '__main__': 
    # define parameters for training 
    training_parameters = {
        'x_obs': th.Tensor([2]),
        'prior': Uniform(-8, 8),
        'n_rounds': 5,
        'n_emulators': 3,
        'train_size': 10000,
        'valid_size': 1000,
        'train_batch_size': 50,
        'train_n_epochs': 250,
        'al_n_samples': 50,
        'al_batch_size': 5,
        'al_n_epochs': 50,
        'lr': .01
    }

    SNLE_ensemble(*training_parameters.values())
