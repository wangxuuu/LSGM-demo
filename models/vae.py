import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# VAE model with linear layers
# used for MNIST
class VAE(nn.Module):
    def __init__(self, input_size=20, h_dim=100, z_dim=10, type='ce'):
        super(VAE, self).__init__()
        self.type = type
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

    def decode(self, z, type='ce'):
        h = F.relu(self.fc4(z))
        if type == 'ce':
            return torch.sigmoid(self.fc5(h))
        elif type == 'mse':
            return self.fc5(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z, self.type)
        return x_reconst, mu, log_var