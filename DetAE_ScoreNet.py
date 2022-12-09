import sys

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
from models import *
from models.utils import *
# from models.scorenetwork import *
# from models.detae import *
from itertools import chain
from torchvision import transforms
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser(description='Determinstic VAE with diffusion model')
parser.add_argument('--config', '-c',
                    dest="filename",
                    help="path to the config file",
                    default='configs/CIFAR_DetAE.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'], name=config['model_params']['name'])

os.CUDA_VISIBLE_DEVICES = config['trainer_params']['gpus']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# random seed
SEED = config['logging_params']['manual_seed']
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

log_path = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
chpt_path = os.path.join(config['logging_params']['model_dir'], config['logging_params']['name'])
sample_path = os.path.join(config['logging_params']['sample_dir'], config['logging_params']['name'])

if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(chpt_path):
    os.makedirs(chpt_path)
if not os.path.exists(sample_path):
    os.makedirs(sample_path)

# load data
# MNIST dataset
if config['data_params']['dataset'] == 'CIFAR10':
    dataset = torchvision.datasets.CIFAR10(root=config['data_params']['data_path'],
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
elif config['data_params']['dataset'] == 'MNIST':
    dataset = torchvision.datasets.MNIST(root=config['data_params']['data_path'],
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
else:
    raise ValueError('Dataset not supported')

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=config['data_params']['train_batch_size'],
                                          shuffle=True)

sigma =  config['exp_params']['sigma']#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


vae = vae_models[config['model_params']['name']](in_channels=3, latent_dim=config['model_params']['latent_dim'], hidden_dims=config['model_params']['hidden_dims'], loss_type=config['model_params']['loss_type']).to(device)
vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config['exp_params']['LR'])

# sn = SN_Model(device, n_steps, sigma_min, sigma_max, dim=z_dim, p = 0.3)
sn = diffusion_models[config['model_params']['diff_name']](marginal_prob_std=marginal_prob_std_fn, embed_dim=config['exp_params']['embed_dim'], dim=config['model_params']['latent_dim'], drop_p=config['exp_params']['drop_p'], num_classes=config['exp_params']['num_classes'], device=device)
sn_optim = torch.optim.Adam(sn.parameters(), lr = config['exp_params']['LR'])
# dynamic = AnnealedLangevinDynamic(sigma_min, sigma_max, n_steps, annealed_step, sn, device, eps=eps)

joint_optim = torch.optim.Adam(params=chain(vae.parameters(), sn.parameters()))

# Start training

for epoch in range(config['trainer_params']['max_epochs']):
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        if config['exp_params']['joint_train']:
            z = vae.encode(x)
            x_reconst = vae.decode(z)
            # Compute reconstruction loss and kl divergence
            # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            vae_losses = vae.loss_function(x_reconst, x)
            reconst_loss = vae_losses['Reconstruction_Loss']
            # reconst_loss = F.mse_loss(x_reconst, x)
            if config['exp_params']['cond']:
                loss_sn = sn.loss(z, y)
            else:
                loss_sn = sn.loss(z)
            loss = loss_sn + reconst_loss

            joint_optim.zero_grad()
            loss.backward()
            joint_optim.step()
        else:
            #============= First Stage: Update VAE ==============#
            # Forward pass
            x_reconst, _, z = vae(x)
            # Compute reconstruction loss and kl divergence
            vae_losses = vae.loss_function(x_reconst, x)
            reconst_loss = vae_losses['Reconstruction_Loss']
            
            vae_optimizer.zero_grad()
            reconst_loss.backward()
            vae_optimizer.step()

            #============= Second Stage: Update SN ==============#
            loss_sn = sn.loss(z.detach())
            vae_optimizer.zero_grad()
            sn_optim.zero_grad()
            loss_sn.backward()
            sn_optim.step()

        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, Diffuse loss: {:.4f}"
                .format(epoch+1, config['trainer_params']['max_epochs'], i+1, len(data_loader), reconst_loss.item(), loss_sn.item()))

    with torch.no_grad():
        # Save the reconstructed images
        out, _, _, = vae(x)
        x_concat = torch.cat([x, out], dim=3)
        save_image(x_concat, os.path.join(sample_path, 'reconst-{}.png'.format(epoch+1)), nrow=4)

        # Save the diffused ima ges
        # dynamic = AnnealedLangevinDynamic(sigma_min, sigma_max, n_steps, annealed_step, sn, device, eps=eps)
        # z_ = forward_proc(z, sigma_min, sigma_max, n_steps, device=device, only_final=True)
        # sample = dynamic.sampling(x.shape[0], z_dim, sample=z_, only_final=True)
        if config['exp_params']['cond']:
            sample = Euler_Maruyama_sampler(sn, marginal_prob_std_fn, diffusion_coeff_fn, dim=config['model_params']['latent_dim'], batch_size=x.shape[0], num_steps=500, device=device, y=y)
        else:
            sample = Euler_Maruyama_sampler(sn, marginal_prob_std_fn, diffusion_coeff_fn, dim=config['model_params']['latent_dim'], batch_size=x.shape[0], num_steps=500, device=device)

        # sample = dynamic.sampling(x.shape[0], z_dim, only_final=True)
        out = vae.decode(sample)
        x_concat = torch.cat([x, out], dim=3)
        save_image(x_concat, os.path.join(sample_path, 'diffuse-{}.png'.format(epoch+1)), nrow=4)
torch.save({'sn_state':sn.state_dict(), 'vae_state':vae.state_dict()}, chpt_path+'ckpt.pth')





