import numpy as np
import torch as th 
from tqdm import trange

def train_network(emulator, 
                  optimizer, 
                  train_dataloader, 
                  valid_dataset, 
                  epochs=100, 
                  print_rate=5, 
                  verbose=True, 
                  progress_bar=True):
    ''' Training description from Lueckmann
        - Adam optimizer (beta_1 = .9, beta_2 = .999)
        - hidden layer with 10 tanh units
        - learning rate = 0.01
    '''
    ## training loop 
    avg_valid_losses, avg_train_losses = [], []
    for epoch in (trange(1, epochs+1) if progress_bar else range(1, epochs+1)):        
        train_loss = []
        for theta, x in train_dataloader:
            optimizer.zero_grad()
            loss = -emulator(theta).log_prob(x).mean()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach())
            # should add a validation loop for early stopping
        train_loss = th.stack(train_loss)
        avg_train_loss, avg_train_std = train_loss.mean().item(), train_loss.std().item()
        avg_train_losses.append([epoch, avg_train_loss, avg_train_std])
        # validation
        with th.no_grad():
            theta_valid, x_valid = valid_dataset
            avg_valid_loss = -emulator(theta_valid.unsqueeze(-1)).log_prob(x_valid).mean().item()
            avg_valid_losses.append([epoch, avg_valid_loss])
        
        if verbose and epoch % print_rate == 0:
                print(f"({epoch}): {avg_train_loss:.3f}", end="")
                print(f", {avg_valid_loss:.3f}")

    return np.array(avg_train_losses), np.array(avg_valid_losses)

'''
def train_network(train_dataloader, valid_dataset, epochs=100, validation_rate=5, lr=.01, verbose=True):
    #Training description from Lueckmann
    #- Adam optimizer (beta_1 = .9, beta_2 = .999)
    #- hidden layer with 10 tanh units
    #- learning rate = 0.01

    emulator = EmulatorNet()
    optimizer = th.optim.Adam(emulator.parameters(), lr=lr)
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in trange(1, epochs+1):        
        train_loss = []
        for theta, x in train_dataloader:
            optimizer.zero_grad()
            loss = -emulator(theta).log_prob(x).mean()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach())
            # should add a validation loop for early stopping
        train_loss = th.stack(train_loss)
        avg_train_loss, avg_train_std = train_loss.mean().item(), train_loss.std().item()
        avg_train_losses.append([epoch, avg_train_loss, avg_train_std])
        
        if epoch % validation_rate == 0:
            if verbose:
                print(f"\n({epoch}): {avg_train_loss:.3f}", end="")
            if valid_dataset is not None:
                with th.no_grad():
                    theta_valid, x_valid = valid_dataset
                    avg_valid_loss = -emulator(theta_valid.unsqueeze(-1)).log_prob(x_valid).mean().item()
                    avg_valid_losses.append([epoch, avg_valid_loss])
                if verbose:
                    print(f", {avg_valid_loss:.3f}", end="")


    return emulator, np.array(avg_train_losses), np.array(avg_valid_losses)
'''


if __name__ == '__main__':
    pass

