import torch.utils.data as data

def create_dataloader(theta, x, batch_size=1):
    trainset = data.TensorDataset(theta.unsqueeze(-1), x)
    dataloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return dataloader