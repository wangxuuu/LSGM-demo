"""
Codebase for the models used in the project.
"""
from .cnnvae import *
from .detae import *
from .diffusion import *
from .nvae import *
from .scorenetwork import *
from .utils import *
from .vae import *

vae_models = {
    'CnnVAE': CnnVAE,
    'DetAE': DetAE,
    'NVAE': NVAE,
    'VAE': VAE
}

diffusion_models = {
    'MLPDiffusion': MLPDiffusion,
    'SN_Model': SN_Model
}