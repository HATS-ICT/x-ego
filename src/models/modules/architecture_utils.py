"""
Architecture utilities for building neural network components.

This module provides common activation functions and building blocks
for constructing neural network architectures.
"""

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    """Quick GELU activation: x * sigmoid(1.702 * x)"""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# Activation function mapping
ACT2CLS = {
    'gelu': nn.GELU,
    'leaky_relu': nn.LeakyReLU,
    'relu': nn.ReLU,
    'linear': nn.Identity,
    'silu': nn.SiLU,
    'quick_gelu': QuickGELU
}


def build_mlp(input_dim, output_dim, num_hidden_layers, hidden_dim, activation='gelu'):
    """
    Build a multi-layer perceptron with specified number of hidden layers.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        num_hidden_layers: Number of hidden layers
        hidden_dim: Dimension of hidden layers
        activation: Name of activation function from ACT2CLS
    
    Returns:
        nn.Sequential: MLP model
    """
    layers = []
    current_dim = input_dim
    
    for i in range(num_hidden_layers):
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            ACT2CLS[activation](),
        ])
        current_dim = hidden_dim
    
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)

