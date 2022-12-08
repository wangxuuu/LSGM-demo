import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

## plot the point data
def scatter(sample, only_final, scatter_range = [-10, 10]):
    # clear_output()
    if only_final:
        scatter = sample.detach().cpu().numpy()
        scatter_x, scatter_y = scatter[:,0], scatter[:,1]
        plt.figure(figsize=(7, 7))

        plt.xlim(scatter_range)
        plt.ylim(scatter_range)
        plt.rc('axes', unicode_minus=False)

        plt.scatter(scatter_x, scatter_y, s=5)
        # plt.show()

    else:
        step_size = sample.size(0)
        fig, axs = plt.subplots(1, step_size, figsize=(step_size * 4, 4), constrained_layout = True)
        for i in range(step_size):
            scatter = sample[i].detach().cpu().numpy()
            scatter_x, scatter_y = scatter[:,0], scatter[:,1]
            axs[i].scatter(scatter_x, scatter_y, s=5)
            axs[i].set_xlim(scatter_range)
            axs[i].set_ylim(scatter_range)
        # plt.show()

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

class Cluster2DataSet(torch.utils.data.Dataset):
    def __init__(self, dist1, dist2, shape = (2), probability=0.2, total_len = 1000000):
        self.dist1_mean, self.dist1_var = dist1[0], dist1[1]
        self.dist2_mean, self.dist2_var = dist2[0], dist2[1]
        self.shape = shape
        self.probability = probability
        self.total_len = total_len

    @property
    def get_probability(self):
        return torch.rand(1) < self.probability

    @property
    def _sampling_1(self):
        return self.dist1_mean + torch.randn(self.shape) * self.dist1_var

    @property
    def _sampling_2(self):
        return self.dist2_mean + torch.randn(self.shape) * self.dist2_var

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self._sampling_1 if self.get_probability else self._sampling_2

        return data

