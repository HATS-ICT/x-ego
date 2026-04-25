import torch
import torch.nn as nn
from einops import repeat

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))  # learned scale for rmsnorm (gamma)

    def forward(self, x):
        # root mean squared norm of x = x / sqrt(mean(x^2) + eps)
        mean_squared = torch.mean(x**2, dim=-1, keepdim=True)
        # torch.rsqrt is 1 / sqrt (faster and numerically stable vs manual 1 / sqrt)
        rms_normed = x * torch.rsqrt(mean_squared + self.eps)
        return rms_normed * self.weight

class SimpleLayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # only center at 0 and make stddev 1 (for film)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)


class AdaptiveNormalizer(nn.Module):
    # either conditioned FiLM or unconditioned RMSNorm
    def __init__(self, embed_dim, conditioning_dim=None):
        super().__init__()
        self.ln = None
        self.rms = None
        self.to_gamma_beta = None
        if conditioning_dim is None:
            # RMSNorm when unconditioned (can do ln but this is better)
            self.rms = RMSNorm(embed_dim)
        else:
            self.ln = SimpleLayerNorm(embed_dim)
            self.to_gamma_beta = nn.Sequential(
                nn.SiLU(),
                nn.Linear(conditioning_dim, 2 * embed_dim)
            )
            # for lam and dynamics we initialize with small non-zero weights 
            # so conditioning is active from step 1 (helps prevent ignoring conditioning)
            nn.init.normal_(self.to_gamma_beta[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, x, conditioning=None):
        # x: [B, T, P, E]
        # conditioning: [B, T, C] or [B, T - 1, C]
        if self.to_gamma_beta is None or conditioning is None:
            normed = self.rms(x) if self.rms is not None else self.ln(x)
            return normed

        x = self.ln(x)
        B, T, P, E = x.shape
        out = self.to_gamma_beta(conditioning) # [B, T, 2 * E]
        out = repeat(out, 'b t twoe -> b t p twoe', p=P) # [B, T, P, 2 * E]
        gamma, beta = out.chunk(2, dim=-1) # each [B, T, P, E]

        # preppend action tensor with zeros in T since a_t-1 should impact z_t (and a_0 is used for z_1 etc)
        if gamma.shape[1] == x.shape[1] - 1 and beta.shape[1] == x.shape[1] - 1:
            gamma = torch.cat([torch.zeros_like(gamma[:, :1]), gamma], dim=1)
            beta = torch.cat([torch.zeros_like(beta[:, :1]), beta], dim=1)

        assert gamma.shape[1] == x.shape[1], f"gamma shape: {gamma.shape} != x shape: {x.shape}"
        assert beta.shape[1] == x.shape[1], f"beta shape: {beta.shape} != x shape: {x.shape}"
        x = x * (1 + gamma) + beta # [B, T, P, E]
        return x