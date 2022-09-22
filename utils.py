import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# VAE model
class VAE(nn.Module):
    def __init__(self, input_size=20, h_dim=100, z_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        # return torch.sigmoid(self.fc5(h))
        return self.fc5(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data = data_root

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, to_cuda=False):
    """Generate samples from a correlated Gaussian distribution."""
    mean = [0,0]
    cov = [[1.0, rho],[rho, 1.0]]
    x, y = np.random.multivariate_normal(mean, cov, batch_size * dim).T

    x = x.reshape(-1, dim)
    y = y.reshape(-1, dim)

    if to_cuda:
        x = torch.from_numpy(x).float().cuda()
        y = torch.from_numpy(y).float().cuda()
        return torch.cat((x, y), dim=1)
    else:
        return np.concatenate((x, y), axis=1)

def plot_fig(X, Y, d=6):
    fig, axs = plt.subplots(d, d, figsize=(8,8), sharex=True, sharey=True)
    for u in range(d):
        for v in range(d):
            axs[u,v].scatter(X[:,u], Y[:,v])
    return fig
