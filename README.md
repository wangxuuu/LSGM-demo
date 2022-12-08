# Lossy Compression with Diffusion model on Latent Space

$$X\rightarrow \underbrace{\text{Encoder}(\cdot)}_{:=q_{\phi}(Z|X)} \rightarrow \underbrace{Z_0 \Leftrightarrow Z_T}_{\text{Diffusion Process}} \rightarrow \underbrace{\text{Decoder}(\cdot)}_{:=p_{\psi}(X|Z)}\rightarrow \hat{X}$$

- Rate: $I(X;\hat{X})$
    - can be approximated by $H(Z_t)$, $Z_t$ is Gaussian distributed.
- Distortion: $E[\Delta(X,\hat{X})]$
    - mse loss.
- Perception: $D(p_X\|p_{\hat{X}})$
    - need to estimate the $f$-divergence based on the samples.

- VAE Loss function:
$$
\begin{align}
\log p(X) &= \log p(X)\int q_{\phi}(z|X)dz\\
&= E_{q_{\phi}(Z|X)}[\log p(X)]\\
&= E_{q_{\phi}(Z|X)}\left[\log \frac{p(X,Z)q_{\phi}(Z|X)}{p(Z|X)q_{\phi}(Z|X)}\right]\\
&= E_{q_{\phi}(Z|X)}\left[\log \frac{p(X,Z)}{q_{\phi}(Z|X)}\right] + E_{q_{\phi}(Z|X)}\left[\log \frac{q_{\phi}(Z|X)}{p(Z|X)}\right]\\
&\geq E_{q_{\phi}(Z|X)}\left[\log \frac{p(X,Z)}{q_{\phi}(Z|X)}\right]\\
&=E_{q_{\phi}(Z|X)}\left[\log \frac{p_{\psi}(X|Z)p(Z)}{q_{\phi}(Z|X)}\right]\\
&= E_{q_{\phi}(Z|X)}[\log p_{\psi}(X|Z)] -E_{q_{\phi}(Z|X)}[\log q_{\phi}(Z|X)] + E_{q_{\phi}(Z|X)}[\log p(Z)]
\end{align}
$$


In original VAE paper [1], the prior distribution $p(Z)$ is chosen as normal distribution $\mathcal{N}(0,I)$. LSGM [2] proposes to model the prior at time $t$ as
$$p(Z_t)\propto \mathcal{N}(Z_t;0,I)^{1-\alpha} p_{\theta}(Z_t)^{\alpha},$$
where $p_{\theta}(Z_t)$ is an intermediate distribution in the denosing diffusion process. The cross entropy term can be written as 
$$
E_{q_{\phi}(Z|X)}[\log p(Z)] = -E\left[\frac{w(t)}{2} E_{q_{\phi}(Z_t,Z_0,t)}[\|\epsilon-\epsilon(Z_t,t)\|^2_2]\right] +c,
$$
where $c$ is a constant and $t\sim U[0,1]$.


## References
[1] Kingma D P, Welling M. Auto-encoding variational bayes[J]. arXiv preprint arXiv:1312.6114, 2013.
[2] Vahdat A, Kreis K, Kautz J. Score-based generative modeling in latent space[J]. Advances in Neural Information Processing Systems, 2021, 34: 11287-11302.

## Codes

- We apply the conditional diffusion model to the latent space of VAE. 

- There are many different settings to train the model, i.e. 
  - Dataset: CIFAR10; MNIST
  - Encoder & decoder : MLP; CNN
  - whether to use the conditional diffusion model
  - diffusion models: Score-based diffusion model; DDPM/DDIM
  - the distribution of the latent variable: Gaussian; delta function (deterministic latent variable)