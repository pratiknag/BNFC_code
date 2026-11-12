# Created by Pratik Nag, August 2025 (approximate)
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.distributions as dist # type: ignore
from torch.distributions import MultivariateNormal # type: ignore
import torch.nn.functional as F # type: ignore
from torch.distributions.normal import Normal # type: ignore
import math
import numpy as np
# from torch_geometric.nn import GCNConv, global_mean_pool
normal = Normal(0.0, 1.0)
class GNN_ENCODER(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=128, output_dim=2):
        super(GNN_ENCODER, self).__init__()

        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim) # type: ignore
        self.conv2 = GCNConv(hidden_dim, hidden_dim) # type: ignore
        self.conv3 = GCNConv(hidden_dim, latent_dim) # type: ignore

        # MLP head for latent mean and log-variance
        self.mean_layer = nn.Linear(latent_dim, output_dim)
        self.logvar_layer = nn.Linear(latent_dim, output_dim)

        self.relu = nn.LeakyReLU(0.2)

    def encode(self, x, edge_index, batch):
        
        # print(x_repeated.shape)
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        # Pool over nodes to get graph-level embedding
        x = global_mean_pool(x,batch)  # type: ignore # [num_graphs, latent_dim]
        # print(x.shape)
        mean = self.mean_layer(x)
        # print(mean.shape)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon.cuda()
        q = dist.Normal(mean, std)
        logp_z = q.log_prob(z)
        return z, logp_z

    def forward(self, x, edge_index, batch):
        mean, logvar = self.encode(x, edge_index, batch)
        std = torch.exp(0.5 * logvar)
        # z, logp_z = self.reparameterization(mean, std)
        return mean,logvar
    

# =========================
# Encoder (Variational Network)
# =========================
class Encoder(nn.Module):
    def __init__(self, n, out):
        super().__init__()

        self.net1 = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU()
            # nn.Linear(128, 64),
            # nn.ReLU()
        )

        self.net2 = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU()
            # nn.Linear(128, 64),
            # nn.ReLU()
        )

        self.mean1 = nn.Linear(64, 1)
        self.logvar1 = nn.Linear(64, 1)

        self.mean2 = nn.Linear(64, out -1 )
        self.logvar2 = nn.Linear(64, out -1)

    def forward(self, x):

        # x: (B,n,1)
        # x = x.squeeze(-1)     # (B,n)

        # combine observations in batch
        x = x.mean(dim=0)     # (n,)

        h1 = self.net1(x)       # (64,)
        h2 = self.net2(x)
        mean = torch.cat([self.mean1(h1), self.mean2(h2)], dim=-1)
        logvar = torch.cat([self.logvar1(h1), self.logvar2(h2)], dim=-1)
        return mean, logvar
    

    
class ElementwisePriorLayer(nn.Module):
    def __init__(self, mean_matrix):
        super().__init__()

        # Store distribution object for weights
        self.weight_prior = mean_matrix

        # Bias prior (vector)
        bias_mean = torch.zeros(
            mean_matrix.size(0),
            device=mean_matrix.device,
            dtype=mean_matrix.dtype
        )
        bias_var = torch.ones(
            mean_matrix.size(0),
            device=mean_matrix.device,
            dtype=mean_matrix.dtype
        ) * 0.01

        self.bias_prior = torch.distributions.Normal(
            loc=bias_mean,
            scale=torch.sqrt(bias_var)
        )

        # Sample once at initialization
        self.weight = nn.Parameter(self.weight_prior)
        self.bias = nn.Parameter(self.bias_prior.sample())

    def forward(self, x):
        return x @ self.weight.T + self.bias


class SimpleNNElementwisePrior(nn.Module):
    def __init__(self, mean_w1, var_w1, mean_w2, var_w2):
        super().__init__()
        self.fc1 = ElementwisePriorLayer(mean_w1, var_w1)
        self.act = nn.ReLU()
        self.fc2 = ElementwisePriorLayer(mean_w2, var_w2)

    def resample_all_priors(self):
        """Resample all weights/biases from their priors."""
        self.fc1.resample_prior()
        self.fc2.resample_prior()

    def forward(self, v):
        h = self.act(self.fc1(v))
        return self.fc2(h)
    

class NNFactor(nn.Module):
    def __init__(self, in_features, hidden1, hidden2):
        """
        in_features: dimension of input vector v = (v1, ..., vk)
        hidden1: hidden size for first NN (processing v2,...,vk)
        hidden2: hidden size for second NN (processing (v1+NN1_output) mod gamma)
        """
        super().__init__()

        # First NN: processes (v2,...,vk)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),   # scalar output with tanh activation
            nn.Linear(hidden2, 1),
            nn.Softplus()
        )

    def forward(self, v):
        """
        v: input tensor of shape (batch, in_features)
        """
        

        out = self.net(v)     # scalar with tanh activation
        return out


# class F_Inv(nn.Module):

#     def __init__(
#         self,
#         theta_dim=3,
#         hidden=128,
#         K=32
#     ):
#         super().__init__()

#         self.K = K

#         self.net = nn.Sequential(
#             nn.Linear(theta_dim, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, hidden),
#             nn.GELU(),
#         )

#         self.shift = nn.Linear(hidden,1)
#         self.scale = nn.Linear(hidden,1)

#         self.alpha = nn.Linear(hidden,K)
#         self.beta = nn.Linear(hidden,K)

#         centers = torch.linspace(-6,6,K)

#         self.register_buffer(
#             "centers",
#             centers
#         )


#         # initialize identity

#         # nn.init.zeros_(self.shift.weight)
#         # nn.init.zeros_(self.shift.bias)

#         # nn.init.zeros_(self.scale.weight)
#         # nn.init.zeros_(self.scale.bias)

#         # nn.init.zeros_(self.alpha.weight)
#         # nn.init.zeros_(self.alpha.bias)

#         # nn.init.zeros_(self.beta.weight)
#         # nn.init.zeros_(self.beta.bias)


#     def transform(self, z, theta):

#         # z:     (...,1)
#         # theta: (...,p)
#         orig_shape = z.shape
        
#         # flatten all batch dimensions
#         z = z.reshape(-1, 1)
#         theta = theta.reshape(-1, theta.shape[-1])


#         h = self.net(theta)


#         a = self.shift(h)          # (N,1)

#         s = torch.exp(
#             self.scale(h)
#         )                           # (N,1)


#         alpha = 0.05 * F.softplus(
#             self.alpha(h)
#         )                           # (N,K)

#         beta = F.softplus(
#             self.beta(h)
#         ) + 1e-4                    # (N,K)



#         # expand z to bins

#         zK = z.unsqueeze(-1).expand(
#             -1,
#             1,
#             self.K
#         )
#         # (N,1,K)


#         centers = self.centers.view(
#             1,1,self.K
#         )


#         alpha = alpha.unsqueeze(1)
#         beta = beta.unsqueeze(1)


#         basis = F.softplus(
#             beta * (zK-centers)
#         )


#         correction = (
#             alpha * basis
#         ).sum(-1)


#         x = (
#             a
#             + s*z
#             + correction
#         )


#         # restore original shape

#         x = x.reshape(orig_shape)

#         return x



#     def forward(self,inputs):

#         """
#         u -> x

#         """

#         u=inputs[...,:1]
#         theta=inputs[...,1:]

        
#         z=torch.logit(
#             u.clamp(1e-6,1-1e-6)
#         )


#         x=self.transform(
#             z,
#             theta
#         )
        
#         return x



#     def inverse(self, x, theta):
#         """
#         x -> u
#         """

#         z = x.clone()

#         for _ in range(30):

#             z.requires_grad_(True)

#             y = self.transform(
#                 z,
#                 theta
#             )

#             dy_dz = torch.autograd.grad(
#                 outputs=y,
#                 inputs=z,
#                 grad_outputs=torch.ones_like(y),
#                 create_graph=True,
#             )[0]

#             z = (
#                 z
#                 - (y - x) / (dy_dz + 1e-6)
#             ).detach()

#         u = torch.sigmoid(z)

#         return u


class F_Inv(nn.Module):

    def __init__(
        self,
        hidden=64,
        K=16,
        beta=2.0,
    ):
        super().__init__()

        self.K = K
        self.beta = beta

        # z -> amplitudes
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.alpha = nn.Linear(hidden, K)

        centers = torch.linspace(-6.0, 6.0, K)
        self.register_buffer("centers", centers)

        # Start as identity transformation
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def transform(self, z):

        orig_shape = z.shape

        z_flat = z.reshape(-1, 1)

        # Condition only on z
        h = self.net(z_flat)

        # Positive amplitudes
        alpha = 0.02 * F.softplus(
            self.alpha(h)
        )                                      # (N, K)

        centers = self.centers.view(1, self.K)

        basis = F.softplus(
            self.beta * (z_flat - centers)
        )                                      # (N, K)

        correction = (alpha * basis).sum(
            dim=-1, keepdim=True
        )

        x = z_flat + correction

        return x.reshape(orig_shape)

    def forward(self, u):

        """
        u in (0,1)
        Returns x = T(logit(u))
        """

        u = u.clamp(1e-6, 1 - 1e-6)

        z = torch.logit(u)

        return self.transform(z)

    @torch.enable_grad()
    def inverse(self, x, n_iter=8):

        """
        x -> z -> u

        Solves T(z) = x using Newton iterations,
        then maps z back to (0,1) using sigmoid.
        """

        x_shape = x.shape

        x = x.reshape(-1, 1)

        # Initial guess
        z = x.clone()

        for _ in range(n_iter):

            z.requires_grad_(True)

            f = self.transform(z)

            df = torch.autograd.grad(
                f,
                z,
                torch.ones_like(f),
                create_graph=False,
            )[0]

            with torch.no_grad():
                z = z - (f - x) / (df + 1e-8)

        # z -> u
        u = torch.sigmoid(z)

        return u.reshape(x_shape)


class NNFactorWithLocations(nn.Module):
    def __init__(self, hidden2, depth=3, out_act="softplus", n_fourier=0, fourier_scale=3.0):
        """
        g(v, s): MLP of (v, s) with `depth` hidden layers of width hidden2 (the default
        reproduces the original architecture and its state-dict keys).
        out_act: "softplus" (g >= 0) or "linear" (both tails of the factor unrestricted).
        n_fourier > 0 appends random Fourier features sin/cos(2 pi B s), B ~ N(0, fourier_scale^2),
        of the coordinates to the input, so that g can vary faster over space.
        """
        super().__init__()
        self.n_fourier = n_fourier
        if n_fourier > 0:
            gen = torch.Generator().manual_seed(0)
            self.register_buffer("fourier_B", fourier_scale * torch.randn(2, n_fourier, generator=gen))
        d_in = 3 + 2 * n_fourier
        layers = [nn.Linear(d_in, hidden2), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden2, hidden2), nn.GELU()]
        layers.append(nn.Linear(hidden2, 1))
        if out_act == "softplus":
            layers.append(nn.Softplus())
        self.net2 = nn.Sequential(*layers)

    def forward(self, v, C):
        """
        v: input tensor of shape (*, in_features) where * can be (batch, h, w, ...)
        gamma: scalar (float or tensor) to take modulo
        C: (num_locations, 2) fixed set of coordinates
        Returns: (*, num_locations)
        """
        batch_shape = v.shape[:-1]        # everything except the feature dimension
        num_locations = C.size(1)
        v = v.unsqueeze(-2)                  # (*, 1, 1)
        v = v.expand(*C.shape[:-1], 1)       # (*, num_locations, 1)
        
        # Concatenate
        inputs = torch.cat([v, C], dim=-1)  # (*, num_locations, 3)
        if self.n_fourier > 0:
            proj = 2 * math.pi * C @ self.fourier_B
            inputs = torch.cat([inputs, torch.sin(proj), torch.cos(proj)], dim=-1)

        
        # Pass through net2
        outputs = self.net2(inputs)              # (prod(*batch_shape)*num_locations, 1)


        # return outputs.squeeze(-1)
        # v: [batch, 1]
        # C: [N, 2]

        # v = v.unsqueeze(-2)                  # [batch, 1, 1]
        # v = v.expand(-1, C.shape[0], -1)     # [batch, N, 1]

        # C = C.unsqueeze(0).expand(v.shape[0], -1, -1)  # [batch, N, 2]

        # inputs = torch.cat([v, C], dim=-1)   # [batch, N, 3]

        # outputs = self.net2(inputs)          # [batch, N, 1]

        return outputs.squeeze(-1)            # [batch, N]
    
def load_F_inv_withlocs(path, device):
    """Loads an F_Inv_WithLocs checkpoint; the two-sided variant is detected from its weights."""
    state = torch.load(path, map_location=device)
    net = F_Inv_WithLocs(coord_dim=2, hidden=128, K=32, two_sided="gamma.weight" in state).to(device)
    net.load_state_dict(state)
    return net


class F_Inv_WithLocs(nn.Module):
    """
    Monotone location-dependent quantile network T(u, s), z = logit(u):

        T = a(s) + s(s) z + sum_k alpha_k(s) softplus{beta_k(s) (z - c_k)}
                          - sum_k gamma_k(s) softplus{-delta_k(s) (z - c_k)}

    with alpha, beta, gamma, delta >= 0. The softplus terms are convex in z;
    with two_sided=True the second sum adds concave terms, so that T can also
    flatten in the tails (light- and moderate-tailed margins), which a convex
    map of z cannot do. Every term is nondecreasing in z, so T stays monotone.
    """

    def __init__(
        self,
        coord_dim=2,
        hidden=128,
        K=32,
        two_sided=False,
        corr_scale=0.05,
        center_range=6.0
    ):
        super().__init__()

        self.K = K
        self.coord_dim = coord_dim
        self.two_sided = two_sided
        self.corr_scale = corr_scale

        # Condition only on spatial coordinates
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.shift = nn.Linear(hidden, 1)
        self.scale = nn.Linear(hidden, 1)

        self.alpha = nn.Linear(hidden, K)
        self.beta = nn.Linear(hidden, K)
        if two_sided:
            self.gamma = nn.Linear(hidden, K)
            self.delta = nn.Linear(hidden, K)
            nn.init.zeros_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

        centers = torch.linspace(-center_range, center_range, K)

        self.register_buffer(
            "centers",
            centers
        )

        # Initialize identity transformation
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)


    def transform(self, z, coords):

        orig_shape = z.shape

        z = z.reshape(-1, 1)
        coords = coords.reshape(-1, coords.shape[-1])

        # Location-specific conditioning
        h = self.net(coords)

        # Location-specific shift
        a = self.shift(h)

        # Location-specific scale
        s = torch.exp(
            self.scale(h)
        )

        # Location-specific nonlinear correction
        alpha = self.corr_scale * F.softplus(
            self.alpha(h)
        )

        beta = F.softplus(
            self.beta(h)
        ) + 1e-4

        # Expand z to K basis functions
        zK = z.unsqueeze(-1).expand(
            -1, 1, self.K
        )

        centers = self.centers.view(
            1, 1, self.K
        )

        alpha = alpha.unsqueeze(1)
        beta = beta.unsqueeze(1)

        basis = F.softplus(
            beta * (zK - centers)
        )

        correction = (
            alpha * basis
        ).sum(-1)

        if self.two_sided:
            gamma = self.corr_scale * F.softplus(self.gamma(h)).unsqueeze(1)
            delta = (F.softplus(self.delta(h)) + 1e-4).unsqueeze(1)
            correction = correction - (
                gamma * F.softplus(-delta * (zK - centers))
            ).sum(-1)

        # Location-specific transformation
        x = (
            a
            + s * z
            + correction
        )

        return x.reshape(orig_shape)


    def forward(self, u, coords):
        """
        u -> x

        u     : (..., 1), values in (0,1)
        coords: (..., 2)
        """

        u = u.clamp(
            1e-6,
            1 - 1e-6
        )

        # u -> z
        z = torch.logit(u)

        # z -> x
        x = self.transform(
            z,
            coords
        )

        return x


    def inverse(self, x, coords, n_iter=30):
        """
        x -> u

        Solves

            transform(z, coords) = x

        and then returns

            u = sigmoid(z)
        """

        x_shape = x.shape

        x = x.reshape(-1, 1)
        coords = coords.reshape(
            -1,
            coords.shape[-1]
        )

        # Initial guess
        z = x.clone()

        for _ in range(n_iter):

            z.requires_grad_(True)

            y = self.transform(
                z,
                coords
            )

            dy_dz = torch.autograd.grad(
                outputs=y,
                inputs=z,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
            )[0]

            z = (
                z
                - (y - x) / (dy_dz + 1e-6)
            ).detach()

        # z -> u
        u = torch.sigmoid(z)

        return u.reshape(x_shape)


class SimpleNNElementwisePriorWithLocations(nn.Module):
    def __init__(self, mean_w1, mean_w2, mean_w3):
        """
        in_features: dimension of input vector v = (v1, ..., vk)
        hidden1: hidden size for first NN (processing v2,...,vk)
        hidden2: hidden size for second NN (processing (modded, location))
        mean_w*, var_w*: prior parameters for elementwise prior layers
        """
        super().__init__()

        # First NN: processes (v2,...,vk) -> scalar with tanh

        # Second NN: processes (modded, C) -> scalar
        self.net2 = nn.Sequential(
            ElementwisePriorLayer(mean_w1),
            nn.Tanh(),
            ElementwisePriorLayer(mean_w2),
            nn.Tanh(),
            ElementwisePriorLayer(mean_w3))

    # def resample_all_priors(self):
    #     """Resample all weights/biases from their priors."""
    #     for m in self.modules():
    #         if isinstance(m, ElementwisePriorLayer):
    #             m.resample_prior()

    def forward(self, v, C):
        """
        v: input tensor of shape (*, in_features) where * can be (batch, h, w, ...)
        gamma: scalar (float or tensor) to take modulo
        C: (num_locations, 2) fixed set of coordinates
        Returns: (*, num_locations)
        """
        batch_shape = v.shape[:-1]        # everything except the feature dimension
        num_locations = C.size(1)
        # v = v.unsqueeze(-2)                  # (*, 1, 1)
        v = v.expand(*C.shape[:-1], 1)       # (*, num_locations, 1)
        
        # Concatenate
        inputs = torch.cat([v, C], dim=-1)  # (*, num_locations, 3)

        
        # Pass through net2
        outputs = self.net2(inputs)              # (prod(*batch_shape)*num_locations, 1)


        return outputs.squeeze(-1)

def rbf_kernel(x1, x2, lengthscale, variance):
    """
    x1, x2: (B, n, 2)
    variance: (B,)
    Returns: (B, n, n)
    """

    # diff = x1.unsqueeze(2) - x2.unsqueeze(1)   # (B, n, n, 2)
    # dist_sq = (diff ** 2).sum(-1)              # (B, n, n)

    sqdist = torch.cdist(x1, x2).pow(2)         # (B, n, n)

    k = variance[:, None, None] * torch.exp(
        -0.5 * sqdist / (lengthscale ** 2)
    )

    # k = variance.unsqueeze(-1).unsqueeze(-1) * \
    #     torch.exp(-0.5 * dist_sq / (lengthscale ** 2))

    return k

def rbf_kernel_sghmc(coords1, coords2, lengthscale, variance):
    """
    RBF kernel without MC dimension (SGHMC-compatible).

    Parameters
    ----------
    coords1 : Tensor (B, n, d)
    coords2 : Tensor (B, n, d)
    lengthscale : float or scalar Tensor
    variance : Tensor (B,)

    Returns
    -------
    K : Tensor (B, n, n)
    """

    # Ensure correct shapes
    B, n, d = coords1.shape

    # Expand for pairwise distances
    x1 = coords1.unsqueeze(2)   # (B, n, 1, d)
    x2 = coords2.unsqueeze(1)   # (B, 1, n, d)

    # Squared Euclidean distance
    sqdist = torch.sum((x1 - x2) ** 2, dim=-1)  # (B, n, n)

    # Reshape variance for broadcasting
    variance = variance.view(B, 1, 1)

    # RBF kernel
    K = variance * torch.exp(-0.5 * sqdist / (lengthscale ** 2))

    return K
    

def gaussian_nll(y_true, mu, cov , eps=1e-5):
    cov = cov + eps * torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    # scale_tril = torch.linalg.cholesky(cov)
    dist = MultivariateNormal(mu, covariance_matrix=cov)
    return dist.log_prob(y_true)

class ExponentialSample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scale, n_samples=50):
        # Sample from a uniform distribution
        eps = 1e-12
        uniform_sample = torch.clamp(torch.rand((n_samples,) + scale.shape, device=scale.device), eps, 1.0)
        
        # Apply the inverse transform sampling method
        exponential_sample = -torch.log(uniform_sample) * (1/scale)
        logprob = torch.log(scale) - (exponential_sample*scale)
        return exponential_sample, logprob

class ParetoSample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, alpha, n_samples=50):
        """
        Sample from shifted Pareto(alpha), support [0, ∞).

        Parameters
        ----------
        alpha : Tensor of shape (...,) 
            Shape parameter (must be > 0).
        n_samples : int
            Number of samples to draw per alpha.

        Returns
        -------
        sample : Tensor of shape (n_samples, *alpha.shape)
        logprob : Tensor of shape (n_samples, *alpha.shape)
        """
        # Sample from Uniform(0,1)
        u = torch.rand((n_samples,) + alpha.shape, device=alpha.device)

        # Pareto with xm=1, then shift so support starts at 0
        raw_sample = (1 - u).pow(-1.0 / alpha)  # (n_samples, *alpha.shape)
        sample = raw_sample - 1.0

        # Log probability: pdf(x) = alpha / (1 + x)^(alpha+1),  x >= 0
        logprob = torch.log(alpha) - (alpha + 1) * torch.log1p(sample)

        return sample, logprob
    
class GammaSample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, shape, rate, num_sample=50):
        # Use PyTorch's distribution object
        gamma_dist = torch.distributions.Gamma(shape, rate)
        sample = gamma_dist.rsample((num_sample,))  # reparameterized sampling
        logprob = gamma_dist.log_prob(sample)

        return sample, logprob

def has_nan_weights(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")
            return True
    return False

def norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def F_exp(y, lam, sigma):
    """
    sigma = Var(Z)
    """
    std = torch.ones_like(sigma)

    return (
        norm_cdf(y / std)
        - torch.exp(
            -lam * y + 0.5 * lam**2 * sigma
        )
        * norm_cdf((y - lam * sigma) / std)
    )


def gamma_pdf(x, alpha, theta):
    return (
        x**(alpha - 1)
        * torch.exp(-x * theta) * theta**alpha
        / (torch.exp(torch.lgamma(alpha)))
    )

def F_gamma(y, theta, sigma, num_samples=100):
    """
    Computes
        P(Z + V <= y)

    Z ~ N(0, sigma)      (sigma = variance)
    V ~ Gamma(alpha, theta) using PyTorch's (concentration, rate)
    parameterization.
    """

    device = y.device
    dtype = y.dtype
    alpha = torch.as_tensor(6.0, device=device, dtype=dtype)
    theta = torch.as_tensor(theta, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)

    gamma_dist = torch.distributions.Gamma(
        concentration=alpha,
        rate=theta,
    )
    v = gamma_dist.sample((num_samples,)).reshape(
        (num_samples,) + (1,) * y.ndim
    )

    std = torch.ones_like(sigma)
    Phi = normal.cdf((y.unsqueeze(0) - v) / std)

    return Phi.mean(dim=0)


def pareto_pdf(x, xm, alpha):
    return alpha * xm**alpha / x**(alpha + 1)

def F_pareto(y, xm, sigma, num_samples=100):
    """
    Computes
        P(Z + V <= y)

    Z ~ N(0, sigma)      (sigma = variance)
    V ~ Pareto(xm + 1, alpha) using PyTorch's (scale, alpha)
    parameterization.
    """

    device = y.device
    dtype = y.dtype
    xm = torch.as_tensor(xm, device=device, dtype=dtype) + 1.0
    alpha = torch.as_tensor(2.0, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)

    pareto_dist = torch.distributions.Pareto(
        scale=xm,
        alpha=alpha,
    )
    v = pareto_dist.sample((num_samples,)).reshape(
        (num_samples,) + (1,) * y.ndim
    )

    std = torch.ones_like(sigma)
    Phi = normal.cdf((y.unsqueeze(0) - v) / std)

    return Phi.mean(dim=0)

def Finv(u, lam, sigma_sq, F, low=-30., high=30., tol=1e-4, max_iter=50):
    u = torch.as_tensor(u)
    low = torch.full_like(u, low)
    high = torch.full_like(u, high)

    lam = torch.as_tensor(lam, device=u.device, dtype=u.dtype)

    for _ in range(max_iter):
        mid = (low + high) / 2
        val = F(mid, lam, sigma_sq)

        low = torch.where(val < u, mid, low)
        high = torch.where(val >= u, mid, high)

        if (high - low).max() < tol:
            break
    # print(F(torch.tensor(low), lam, sigma_sq))
    # print(F(torch.tensor(high), lam, sigma_sq))
    # print(u.min(), u.max())
    return (low + high) / 2



def log_integrated_density(
    x,
    Sigma,
    gamma,
    factor,
    num_quad=100,
    upper=8.0,
):
    """
    Computes

        log ∫ φΣ(x-v1) fV(v) dv

    Parameters
    ----------
    x : (n,)
        One observation.
    Sigma : (n,n)
        GP covariance matrix.
    gamma : scalar tensor
    factor : {"exponential","gamma","pareto"}
    num_quad : int
        Number of quadrature points.
    upper : float
        Truncation point for infinite integrals.

    Returns
    -------
    scalar tensor
    """

    device = x.device
    dtype = x.dtype
    N, n = x.shape

    mvn = dist.MultivariateNormal(
        loc=torch.zeros_like(x, device=device, dtype=dtype),
        covariance_matrix=Sigma
    )

    if factor == "exponential":

        grid = torch.linspace(
            0.0,
            upper,
            num_quad,
            device=device,
            dtype=dtype,
        )

        prior = dist.Exponential(gamma)

    elif factor == "gamma":

        grid = torch.linspace(
            0.0,
            upper,
            num_quad,
            device=device,
            dtype=dtype,
        )

        prior = dist.Gamma(
            gamma + 1e-4,
            torch.tensor(
                1.0,
                device=device,
                dtype=dtype,
            ),
        )

    elif factor == "pareto":

        grid = torch.linspace(
            1.0,
            upper,
            num_quad,
            device=device,
            dtype=dtype,
        )

        prior = dist.Pareto(
            gamma + 1.0,
            torch.tensor(
                1.0,
                device=device,
                dtype=dtype,
            ),
        )

    else:
        raise ValueError("Unknown factor.")

    # grid spacing
    delta = grid[1] - grid[0]

    # Compute log integrand
    log_terms = []

    ones = torch.ones_like(x, device=device, dtype=dtype)

    for v in grid:

        log_normal = mvn.log_prob(
            x - v * ones
        )
        
        log_prior = prior.log_prob(v)
        log_terms.append(log_normal + log_prior)

    log_terms = torch.stack(log_terms)
    #
    # log ∫ exp(log_terms) dv
    #
    return torch.logsumexp(log_terms, dim=0) + torch.log(delta)


def log_marginal_density(
    w,
    gamma,
    factor,
    num_quad=100,
    upper=8.0,
):
    """
    Computes

        log f1(w)
        = log ∫ φ(w-v) fV(v) dv

    Parameters
    ----------
    w : scalar tensor
    gamma : scalar tensor
    factor : {"exponential", "gamma", "pareto"}
    """

    device = w.device
    dtype = w.dtype
    std = torch.sqrt(torch.tensor(1.0, device=device, dtype=dtype))
    normal = dist.Normal(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )

    if factor == "exponential":

        grid = torch.linspace(
            0.0,
            upper,
            num_quad,
            device=device,
            dtype=dtype,
        )

        prior = dist.Exponential(gamma)

    elif factor == "gamma":

        grid = torch.linspace(
            0.0,
            upper,
            num_quad,
            device=device,
            dtype=dtype,
        )

        prior = dist.Gamma(
            gamma + 1e-4,
            torch.tensor(1.0, device=device, dtype=dtype),
        )

    elif factor == "pareto":

        grid = torch.linspace(
            1.0,
            upper,
            num_quad,
            device=device,
            dtype=dtype,
        )

        prior = dist.Pareto(
            gamma + 1.0,
            torch.tensor(1.0, device=device, dtype=dtype),
        )

    else:
        raise ValueError("Unknown factor")

    delta = grid[1] - grid[0]

    log_terms = []
    ones = torch.ones_like(w, device=device, dtype=dtype)
    for v in grid:

        log_normal = normal.log_prob((w - v * ones)/std)

        log_prior = prior.log_prob(v)

        log_terms.append(log_normal + log_prior)

    log_terms = torch.stack(log_terms)

    return torch.logsumexp(log_terms, dim=0) + torch.log(delta)


def main():
    # ===== Example usage SimpleNN =====

    # # Define elementwise means/variances for each layer
    # mean_w1 = torch.zeros(10, 1)       # layer1: 10 hidden, 1 input
    # var_w1  = torch.ones(10, 1) * 0.1

    # mean_w2 = torch.zeros(1, 10)       # layer2: 1 output, 10 hidden
    # var_w2  = torch.ones(1, 10) * 0.05

    # # Create model
    # model = SimpleNNElementwisePrior(mean_w1, var_w1, mean_w2, var_w2)

    # # Forward pass
    # V = torch.rand(5, 1)  # V ~ U(0,1)
    # output = model(V)
    # print("Output:\n", output)

    # # Resample weights/biases from priors
    # model.resample_all_priors()
    # output2 = model(V)
    # print("Output after resampling:\n", output2)

    # ===== Example usage Factor models =====
    model = NNFactor(in_features=5, hidden1=16, hidden2=8)
    v = torch.randn(10, 5)   # batch of 10, input dimension = 5
    gamma = 3.14             # example gamma
    output = model(v, gamma)
    print(output.shape)  # torch.Size([10, 1])
    print(output)
    model = NNFactorWithLocations(in_features=5, hidden1=16, hidden2=30)
    v = torch.randn(10, 5)   # batch of 10, input dimension = 5
    C = torch.randn(25, 2)      # locations
    gamma = torch.rand(10) * 3  # batch of gammas
    output = model(v, gamma, C)
    print(output.shape)  # torch.Size([10, 1])
    print(output)






# =========================
# Vecchia approximation of the Gaussian component
# =========================
def maxmin_order(coords):
    """
    Maximin ordering of the locations (Guinness, 2018): start at the location
    closest to the centre, then repeatedly add the location farthest from those
    already ordered. coords: (n, 2) numpy array. Returns a permutation (n,).
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    first = int(np.argmin(((coords - coords.mean(0)) ** 2).sum(1)))
    order = [first]
    dmin = np.sqrt(((coords - coords[first]) ** 2).sum(1))
    dmin[first] = -np.inf
    for _ in range(n - 1):
        k = int(np.argmax(dmin))
        order.append(k)
        dmin = np.minimum(dmin, np.sqrt(((coords - coords[k]) ** 2).sum(1)))
        dmin[k] = -np.inf
    return np.array(order)


def vecchia_neighbours(coords_ordered, m):
    """
    Conditioning sets of the Vecchia approximation: for the i-th location in the
    ordering, the indices of its min(m, i) nearest locations among the first i.
    Returns an (n, m + 1) index array whose first column is i itself, padded
    with -1 where i < m.
    """
    c = np.asarray(coords_ordered, dtype=np.float64)
    n = len(c)
    nbr = np.full((n, m + 1), -1, dtype=np.int64)
    nbr[:, 0] = np.arange(n)
    for i in range(1, n):
        d = ((c[:i] - c[i]) ** 2).sum(1)
        k = min(m, i)
        nbr[i, 1:k + 1] = np.argpartition(d, k - 1)[:k] if k < i else np.arange(i)
    return nbr


class VecchiaFactor:
    """
    Sparse inverse-Cholesky (Vecchia) factor U of a correlation matrix Sigma,
    with Sigma^{-1} ~ U U' (Schafer, Katzfuss and Owhadi, 2021). Column i of U is
    nonzero on the conditioning set K_i = {i} + N(i):

        U[K_i, i] = Sigma[K_i, K_i]^{-1} e_i / sqrt(e_i' Sigma[K_i, K_i]^{-1} e_i),

    computed from the (m + 1) x (m + 1) blocks only, so that the cost is
    O(n m^3) and Sigma is never formed. The blocks are built from the kernel, so
    U is differentiable in its range and nugget. For a residual r (in the
    original order of the locations),

        r' Sigma^{-1} r ~ |U' r|^2,   log|Sigma| ~ -2 sum_i log U_ii,

    at cost O(n m) per residual.
    """

    def __init__(self, coords, m, device="cpu"):
        coords = np.asarray(coords, dtype=np.float64)
        self.n, self.m = len(coords), m
        self.order = maxmin_order(coords)
        nbr = vecchia_neighbours(coords[self.order], m)
        self.mask = torch.tensor(nbr >= 0, device=device)                     # (n, m+1)
        # indices into the ORIGINAL ordering of the locations
        orig = np.where(nbr >= 0, self.order[np.clip(nbr, 0, None)], 0)
        self.idx = torch.tensor(orig, dtype=torch.long, device=device)        # (n, m+1)
        self.coords = torch.tensor(coords, device=device)

    def factor(self, kernel="se", lengthscale=0.1, nugget=1e-4, dtype=torch.float64):
        """Values of U on the conditioning sets, (n, m+1); column 0 is U_ii."""
        c = self.coords.to(dtype)[self.idx]                                   # (n, m+1, 2)
        d = torch.cdist(c, c)
        ls = torch.as_tensor(lengthscale, dtype=dtype, device=c.device)
        tau = torch.as_tensor(nugget, dtype=dtype, device=c.device)
        if kernel == "fixed":                       # SE, range 0.1, nugget 1e-4 added
            K = torch.exp(-0.5 * d ** 2 / 0.1 ** 2)
            K = K + 1e-4 * torch.eye(self.m + 1, dtype=dtype, device=c.device)
        else:
            K = torch.exp(-d / ls) if kernel == "exponential" else torch.exp(-0.5 * d ** 2 / ls ** 2)
            K = (1 - tau) * K + tau * torch.eye(self.m + 1, dtype=dtype, device=c.device)
        # padded entries: decouple them (identity rows/columns)
        mk = self.mask.to(c.device)
        pair = mk[:, :, None] & mk[:, None, :]
        eye = torch.eye(self.m + 1, dtype=dtype, device=c.device).expand_as(K)
        K = torch.where(pair, K, eye)
        e = torch.zeros(self.n, self.m + 1, 1, dtype=dtype, device=c.device)
        e[:, 0] = 1.0
        v = torch.cholesky_solve(e, torch.linalg.cholesky(K))[..., 0]          # (n, m+1)
        U = v / torch.sqrt(v[:, :1])
        return torch.where(mk, U, torch.zeros_like(U))

    def quad_logdet(self, r, U):
        """
        r: (..., n) residuals in the original order. Returns (|U' r|^2 (...), log|Sigma|).
        z = U' r is computed as a sparse-dense product, z_i = sum_j U[i, j] r[idx[i, j]],
        so that memory is O(rows x n) rather than O(rows x n x m); gradients flow to
        both r and U (and hence to the range and nugget).
        """
        mk = self.mask.to(U.device)
        rows = torch.arange(self.n, device=U.device)[:, None].expand_as(self.idx)[mk]
        Usp = torch.sparse_coo_tensor(torch.stack((rows, self.idx.to(U.device)[mk])), U[mk],
                                      (self.n, self.n))
        flat = r.reshape(-1, self.n)
        z = torch.sparse.mm(Usp, flat.T).T.reshape(r.shape)                  # (..., n)
        return (z ** 2).sum(-1), -2.0 * torch.log(U[:, 0]).sum()

    def log_prob(self, r, U):
        """Gaussian log-density of the residuals r (..., n) under the approximation."""
        quad, logdet = self.quad_logdet(r, U)
        return -0.5 * quad - 0.5 * logdet - 0.5 * self.n * math.log(2 * math.pi)

if __name__ == '__main__':
    main()

