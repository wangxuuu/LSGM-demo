import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPDiffusion(nn.Module):
    def __init__(self,n_steps, dim=20, num_units=128, num_classes=None):
        '''
        num_units: hidden size
        '''
        super(MLPDiffusion,self).__init__()
        self.num_classes = num_classes
        self.linears = nn.ModuleList(
            [
                nn.Linear(dim,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,dim),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, num_units)

    def forward(self,x,t, y=None):
#         x = x_0
        # conditioning on label
        if y is not None:
            assert y.shape == (x.shape[0],)
            class_emb = self.label_emb(y)

        for idx,embedding_layer in enumerate(self.step_embeddings):
            embedding = embedding_layer(t)
            if y is not None:
                embedding = embedding + class_emb
            x = self.linears[2*idx](x)
            x += embedding
            x = self.linears[2*idx+1](x)

        x = self.linears[-1](x)

        return x

def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps, y=None):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    #对一个batchsize样本生成随机的时刻t
    t = torch.randint(0,n_steps,size=(batch_size//2,))
    t = torch.cat([t,n_steps-1-t],dim=0)
    t = t.unsqueeze(-1).to(device)

    #x0的系数
    a = alphas_bar_sqrt[t]

    #eps的系数
    aml = one_minus_alphas_bar_sqrt[t]

    #生成随机噪音eps
    e = torch.randn_like(x_0)

    #构造模型的输入
    x = x_0*a+e*aml

    #送入模型，得到t时刻的随机噪声预测值
    output = model(x,t.squeeze(-1),y)

    #与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()

def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.Tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

## DDIM sampling
def ddim_sample(model, num_steps=100, batch_size=64, dim=20, x_T=None, y=None):
    """
    Sample x_0 from x_t using DDIM, assuming a method on predictor called
    predict_epsilon(x_t, alphas).
    """
    def create_alpha_schedule(num_steps=100, beta_0=0.0001, beta_T=0.02):
        betas = np.linspace(beta_0, beta_T, num_steps)
        result = [1.0]
        alpha = 1.0
        for beta in betas:
            alpha *= 1 - beta
            result.append(alpha)
        return torch.Tensor(result).to(device)

    def sample_q(x_0, ts, epsilon=None):
        """
        Sample from q(x_t | x_0) for a batch of x_0.
        """
        if epsilon is None:
            epsilon = torch.randn(size=x_0.shape)
        alphas = alphas_for_ts(ts, x_0.shape)
        return torch.sqrt(alphas) * x_0 + torch.sqrt(1 - alphas) * epsilon

    def predict_x0(x_t, ts, epsilon_prediction):
        alphas = alphas_for_ts(ts, x_t.shape)
        return (x_t - torch.sqrt(1 - alphas) * epsilon_prediction) / torch.sqrt(alphas)

    def ddim_previous(x_t, ts, epsilon_prediction):
        """
        Take a ddim sampling step given x_t, t, and epsilon prediction.
        """
        x_0 = predict_x0(x_t, ts, epsilon_prediction)
        return sample_q(x_0, ts - 1, epsilon=epsilon_prediction)

    def alphas_for_ts(ts, shape=None):
        alphas = alpha[ts]
        if shape is None:
            return alphas
        while len(alphas.shape) < len(shape):
            alphas = alphas[..., None]
        return alphas

    if x_T is None:
        x_T = torch.randn(batch_size, dim, device=device)

    alpha = create_alpha_schedule(num_steps)
    x_t = x_T
    t_iter = range(1, num_steps)[::-1]

    for t in t_iter:
        ts = torch.tensor([t] * x_T.shape[0]).to(device)
        # alphas = alphas_for_ts(ts)
        x_t = ddim_previous(x_t, ts, model(x_t, ts, y))
    return x_t

