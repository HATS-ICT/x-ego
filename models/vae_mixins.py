"""
VAE Components for Generative Multi-Agent Location Prediction

This module provides VAE-specific functionality including:
- Encoder (posterior approximation)
- Decoder (generative model)
- Reparameterization trick
- Prior sampling
"""

import torch


class VAEMixin:
    """Mixin class providing VAE functionality for generative models."""
    
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
        if self.task_form != 'generative':
            raise ValueError("encode() only available in generative mode")
        
        # Flatten target locations if needed
        if target_locations.dim() == 3:  # [B, 5, 3]
            target_locations = target_locations.view(target_locations.shape[0], -1)
        
        # Concatenate with conditioning
        encoder_input = torch.cat([target_locations, combined_features], dim=1)
        
        # Get mu and logvar
        encoder_output = self.vae_encoder(encoder_input)
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
        if self.task_form != 'generative':
            raise ValueError("decode() only available in generative mode")
        
        # Concatenate latent with conditioning
        decoder_input = torch.cat([z, combined_features], dim=1)
        
        # Decode to predictions
        predictions = self.predictor(decoder_input)
        predictions = predictions.view(-1, 5, 3)
        return predictions
    
    def sample_from_prior(self, combined_features, num_samples=1):
        """
        Sample from prior distribution for generative testing.
        
        Args:
            combined_features: [B, combined_dim] - conditioning features
            num_samples: int - number of samples to generate per conditioning
            
        Returns:
            samples: [B, num_samples, 5, 3] - generated samples
        """
        if self.task_form != 'generative':
            raise ValueError("sample_from_prior() only available in generative mode")
        
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
    
    @torch.inference_mode()
    def generate_multiple_predictions(self, sample, num_predictions=100):
        """
        Generate multiple predictions for a single test sample.
        
        Args:
            sample: Dictionary containing 'video', 'pov_team_side_encoded', 'enemy_locations'
            num_predictions: Number of predictions to generate
            
        Returns:
            predictions: numpy array of shape [num_predictions, 5, 3] (unscaled)
            target: numpy array of shape [5, 3] (unscaled ground truth)
        """
        self.eval()
        
        predictions = []
        for _ in range(num_predictions):
            if self.task_form == 'generative':
                outputs = self.forward(sample, mode='sampling')
            else:
                outputs = self.forward(sample, mode='full')
            
            pred = outputs['predictions']  # [1, 5, 3]
            
            if self.task_form in ['coord-reg', 'generative']:
                pred_unscaled = self.unscale_coordinates(pred)
                predictions.append(pred_unscaled.cpu().numpy()[0])
            else:
                raise NotImplementedError(
                    "Multiple predictions generation only supported for coordinate-based tasks"
                )
        
        predictions = torch.tensor(predictions).numpy()  # [num_predictions, 5, 3]
        
        # Get unscaled ground truth
        target_unscaled = self.unscale_coordinates(sample['enemy_locations'])
        target = target_unscaled.cpu().numpy()[0]  # [5, 3]
        
        return predictions, target
