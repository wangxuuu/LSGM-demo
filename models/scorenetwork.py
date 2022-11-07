# Score-based diffusion model
import torch
import math
import torch.nn as nn
import functools
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class SN_Model(nn.Module):
#     def __init__(self, device, n_steps, sigma_min, sigma_max, dim, p=0.5):
#         '''
#         Score Network.

#         n_steps   : perturbation schedule steps (Langevin Dynamic step)
#         sigma_min : sigma min of perturbation schedule
#         sigma_min : sigma max of perturbation schedule

#         '''
#         super().__init__()
#         self.device = device
#         self.sigmas = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps)).to(device = device)

#         self.linear_model1 = nn.Sequential(
#             nn.Linear(dim, 256),
#             nn.Dropout(p),
#             nn.GELU(),
#         )
#         # Condition sigmas
#         self.embedding_layer = nn.Embedding(n_steps, 256)

#         self.linear_model2 = nn.Sequential(
#             nn.Linear(256, 512),
#             nn.Dropout(p),
#             nn.GELU(),

#             nn.Linear(512, 512),
#             nn.Dropout(p),
#             nn.GELU(),

#             nn.Linear(512, dim),
#         )

#         self.to(device = self.device)


#     def loss_fn(self, x, idx=None):
#         '''
#         This function performed when only training phase.

#         x          : real data if idx==None else perturbation data
#         idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.

#         '''
#         scores, target, sigma = self.forward(x, idx=idx, get_target=True)

#         target = target.view(target.shape[0], -1)
#         scores = scores.view(scores.shape[0], -1)


#         losses = torch.square(scores - target).mean(dim=-1) * sigma.squeeze() ** 2
#         return losses.mean(dim=0)


#     def forward(self, x, idx=None, get_target=False):
#         '''
#         x          : real data if idx==None else perturbation data
#         idx        : if None (training phase), we perturbed random index. Else (inference phase), it is recommended that you specify.
#         get_target : if True (training phase), target and sigma is returned with output (score prediction)
#         '''
#         if idx == None:
#             idx = torch.randint(0, len(self.sigmas), (x.size(0),)).to(device = self.device)
#             used_sigmas = self.sigmas[idx][:,None]
#             noise = torch.randn_like(x)
#             x_tilde = x + noise * used_sigmas

#         else:
#             idx = torch.cat([torch.Tensor([idx for _ in range(x.size(0))])]).long().to(device = self.device)
#             used_sigmas = self.sigmas[idx][:,None]
#             x_tilde = x

#         if get_target:
#             target = - 1 / used_sigmas * noise

#         output = self.linear_model1(x_tilde)
#         embedding = self.embedding_layer(idx)
#         output = self.linear_model2(output + embedding)

#         return (output, target, used_sigmas) if get_target else output

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        # x is a vector of time steps. The dim of the results is (len(x), embed_dim).
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)

def marginal_prob_std(t, sigma=25):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE. a hyper-parameter

    Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)

# sigma =  25.0#@param {'type':'number'}
# marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
# diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

class SN_Model(nn.Module):
    def __init__(self, marginal_prob_std, embed_dim, dim, drop_p=0.3, device='cpu'):
        super().__init__()
        self.device = device
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        self.embedding_layer = Dense(embed_dim, 256)
        self.linear_model1 = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Dropout(drop_p),
            nn.GELU(),
        )
        # Condition sigmas
        # self.embedding_layer = nn.Embedding(embed_dim, 256)

        self.linear_model2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(drop_p),
            nn.GELU(),

            nn.Linear(512, 512),
            nn.Dropout(drop_p),
            nn.GELU(),

            nn.Linear(512, dim),
        )

        self.to(device = self.device)

    def forward(self, x, t):
        embed = self.act(self.embed(t))
        output = self.linear_model1(x)
        output = self.linear_model2(output + self.embedding_layer(embed))
        return output/self.marginal_prob_std(t)[:, None]
        
    def loss(self, x, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
            model: A PyTorch model instance that represents a 
            time-dependent score-based model.
            x: A mini-batch of training data.    
            marginal_prob_std: A function that gives the standard deviation of 
            the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)
        perturbed_x = x + z * std[:, None]
        score = self.forward(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None] + z).reshape(x.shape[0], -1)**2, dim=1))
        return loss


# ## LangevinDynamic Sampling
# class AnnealedLangevinDynamic():
#     def __init__(self, sigma_min, sigma_max, n_steps, annealed_step, score_fn, device, eps = 1e-1):
#         '''
#         sigma_min       : minimum sigmas of perturbation schedule
#         sigma_max       : maximum sigmas of perturbation schedule
#         n_steps         : iteration step of Langevin dynamic
#         annealed_step   : annelaed step of annealed Langevin dynamic
#         score_fn        : trained score network
#         eps             : coefficient of step size
#         '''

#         self.process = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps))
#         self.step_size = eps * (self.process / self.process[-1] ) ** 2
#         self.score_fn = score_fn
#         self.annealed_step = annealed_step
#         self.device = device

#     def _one_annealed_step_iteration(self, x, idx):
#         '''
#         x   : perturbated data
#         idx : step of perturbation schedule
#         '''

#         self.score_fn.eval()
#         z, step_size = torch.randn_like(x).to(device = self.device), self.step_size[idx] # z: noise
#         x = x + 0.5 * step_size * self.score_fn(x, idx) + torch.sqrt(step_size) * z
#         return x

#     def _one_annealed_step(self, x, idx):
#         '''
#         x   : perturbated data
#         idx : step of perturbation schedule
#         '''
#         for _ in range(self.annealed_step):
#             x = self._one_annealed_step_iteration(x, idx)
#         return x

#     def _one_perturbation_step(self, x):
#         '''
#         x   : sampling of prior distribution
#         '''
#         for idx in range(len(self.process)):
#             x = self._one_annealed_step(x, idx)
#             yield x

#     @torch.no_grad()
#     def sampling(self, sampling_number, dim, sample=None, only_final=False):
#         '''
#         only_final : If True, return is an only output of final schedule step
#         '''
#         if sample is None:
#             sample = (torch.rand([sampling_number,dim]).to(device = self.device) - 0.5)*2
#         sampling_list = []

#         final = None
#         for sample in self._one_perturbation_step(sample):
#             final = sample
#             if not only_final:
#                 sampling_list.append(final)


#         return final if only_final else torch.stack(sampling_list)



# the forward process in SN_model
@torch.no_grad()
def forward_process(x,
                    marginal_prob_std,
                    batch_size=64,
                    num_steps=10,
                    device='cpu',
                    eps=1e-3,
                    end_t=1.,
                    only_final=True):
    sampling_list = []
    time_steps = torch.linspace(eps, end_t, num_steps, device=device)
    if only_final:
        return x + marginal_prob_std(time_steps[-1]) * torch.randn_like(x)
    else:
        for time_step in time_steps:
            sampling_list.append(x + marginal_prob_std(time_step) * torch.randn_like(x))
        return torch.stack(sampling_list)

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
# num_steps =  500#@param {'type':'integer'}
from mimetypes import init

#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.

def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           dim=2,
                           batch_size=64, 
                           num_steps=100, 
                           device='cpu', 
                           eps=1e-3,
                           start_t=1.,
                           save_times=8,
                           only_final=True,
                           init_x=None):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.
    
    Returns:
        Samples.    
    """
    sampling_list = []
    t = torch.ones(batch_size, device=device)
    # set t=1 to approximate the marginal distribution, init_x
    if init_x is None:
        init_x = torch.randn(batch_size, dim, device=device) \
            * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(start_t, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    i = 0
    with torch.no_grad():
        for time_step in time_steps:      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)  
            if not only_final and i%(num_steps//save_times)==0:
                sampling_list.append(x)
            i+=1   
    # Do not include any noise in the last sampling step.
    return mean_x if only_final else torch.stack(sampling_list)
