"""
VAE Components for coord-gen Multi-Agent Location Prediction

This module provides VAE-specific functionality including:
- Encoder (posterior approximation)
- Decoder (coord-gen model)
- Reparameterization trick
- Prior sampling
"""

import torch
import torch.nn as nn


class ConditionalVariationalAutoencoder(nn.Module):
    """VAE module for coord-gen multi-agent location prediction."""
    
    def __init__(self, output_dim, combined_dim, latent_dim, num_hidden_layers, 
                 hidden_dim, dropout, activation_fn):
        """
        Initialize VAE encoder and decoder networks.
        
        Args:
            output_dim: Dimension of target locations (e.g., 5 * 3 = 15)
            combined_dim: Dimension of conditioning features
            latent_dim: Dimension of latent space
            num_hidden_layers: Number of hidden layers in encoder/decoder
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            activation_fn: Activation function class
        """
        super().__init__()
        self.output_dim = output_dim
        self.combined_dim = combined_dim
        self.latent_dim = latent_dim
        
        # Encoder: target_locations + conditioning -> mu, logvar
        encoder_input_dim = output_dim + combined_dim
        self.encoder = self._build_mlp(
            encoder_input_dim, latent_dim * 2, 
            num_hidden_layers, hidden_dim, dropout, activation_fn
        )
        
        # Decoder: latent + conditioning -> predictions
        decoder_input_dim = latent_dim + combined_dim
        self.decoder = self._build_mlp(
            decoder_input_dim, output_dim, 
            num_hidden_layers, hidden_dim, dropout, activation_fn
        )
    
    def _build_mlp(self, input_dim, output_dim, num_hidden_layers, hidden_dim, 
                   dropout, activation_fn):
        """Build a multi-layer perceptron."""
        layers = []
        current_dim = input_dim
        
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
    
    def encode(self, target_locations, combined_features):
        """
        VAE encoder: encode target locations + conditioning to latent distribution parameters.
        
        Args:
            target_locations: [B, 5, 3] or [B, 15] - target enemy locations
            combined_features: [B, combined_dim] - conditioning features
            
        Returns:
            mu: [B, latent_dim] - mean of latent distribution
            logvar: [B, latent_dim] - log variance of latent distribution
        """
        # Flatten target locations if needed
        if target_locations.dim() == 3:  # [B, 5, 3]
            target_locations = target_locations.view(target_locations.shape[0], -1)
        
        # Concatenate with conditioning
        encoder_input = torch.cat([target_locations, combined_features], dim=1)
        
        # Get mu and logvar
        encoder_output = self.encoder(encoder_input)
        mu, logvar = torch.chunk(encoder_output, 2, dim=1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        VAE reparameterization trick: sample from N(mu, exp(logvar/2)).
        
        Args:
            mu: [B, latent_dim] - mean
            logvar: [B, latent_dim] - log variance
            
        Returns:
            z: [B, latent_dim] - sampled latent code
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z, combined_features):
        """
        VAE decoder: decode latent + conditioning to predictions.
        
        Args:
            z: [B, latent_dim] - latent code
            combined_features: [B, combined_dim] - conditioning features
            
        Returns:
            predictions: [B, 5, 3] - predicted enemy locations
        """
        # Concatenate latent with conditioning
        decoder_input = torch.cat([z, combined_features], dim=1)
        
        # Decode to predictions
        predictions = self.decoder(decoder_input)
        predictions = predictions.view(-1, 5, 3)
        return predictions
    
    def sample_from_prior(self, combined_features, num_samples=1):
        """
        Sample from prior distribution for coord-gen testing.
        
        Args:
            combined_features: [B, combined_dim] - conditioning features
            num_samples: int - number of samples to generate per conditioning
            
        Returns:
            samples: [B, num_samples, 5, 3] - generated samples
        """
        batch_size = combined_features.shape[0]
        device = combined_features.device
        
        samples = []
        for _ in range(num_samples):
            # Sample from standard normal prior
            z = torch.randn(batch_size, self.latent_dim, device=device)
            # Decode with conditioning
            sample = self.decode(z, combined_features)
            samples.append(sample)
        
        samples = torch.stack(samples, dim=1)  # [B, num_samples, 5, 3]
        return samples
    
    def forward(self, target_locations, combined_features, mode='full'):
        """
        Forward pass through VAE.
        
        Args:
            target_locations: [B, 5, 3] - target locations (for training)
            combined_features: [B, combined_dim] - conditioning features
            mode: 'full' for encode-decode, 'sampling' for prior sampling
            
        Returns:
            Dictionary with predictions and latent variables
        """
        if mode == 'sampling':
            predictions = self.sample_from_prior(combined_features, num_samples=1)
            predictions = predictions.squeeze(1)  # [B, 5, 3]
            return {
                'predictions': predictions,
                'mu': None,
                'logvar': None,
                'z': None
            }
        else:
            mu, logvar = self.encode(target_locations, combined_features)
            z = self.reparameterize(mu, logvar)
            predictions = self.decode(z, combined_features)
            return {
                'predictions': predictions,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
    
