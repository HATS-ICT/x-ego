import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
import torch._dynamo
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import numpy as np
import json
from pathlib import Path
from geomloss import SamplesLoss
from scipy.stats import wasserstein_distance


from utils.plot_utils import create_prediction_plots, create_prediction_heatmaps_grid
from utils.serialization_utils import json_serializable
from utils.metric_utils import multinomial_loss, exact_match_accuracy, l1_count_error, kl_divergence_histogram, chamfer_distance_batch
try:
    from video_encoder_baseline import VideoEncoderBaseline
except ImportError:
    from .video_encoder_baseline import VideoEncoderBaseline


class MultiAgentEnemyLocationPredictionModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.video_encoder = self.init_video_encoder(config['model']['encoder']['video'])
        self.num_agents = config['data']['num_agents']
        self.task_form = config['data']['task_form']
        
        # Team encoder: embed team side (T=0, CT=1) into a vector
        self.team_embed_dim = config['model']['team_embed_dim']  # Small embedding dimension for team side
        self.team_encoder = nn.Embedding(2, self.team_embed_dim)  # 2 teams: T, CT
        
        self.loss_fn = config['data']['loss_fn']
        self.sinkhorn_blur = config['data']['sinkhorn_blur']
        self.sinkhorn_scaling = config['data']['sinkhorn_scaling']
        
        self.agent_fusion_method = config['model']['agent_fusion_method']  # 'mean', 'max', 'attention', 'concat'
        
        if self.agent_fusion_method == 'attention':
            self.agent_attention = nn.MultiheadAttention(
                embed_dim=self.video_encoder.embed_dim,
                num_heads=8,
                batch_first=True
            )
        elif self.agent_fusion_method == 'concat':
            self.agent_fusion_proj = nn.Linear(
                self.video_encoder.embed_dim * self.num_agents, 
                self.video_encoder.embed_dim
            )
        hidden_dim = config['model']['hidden_dim']
        dropout = config['model']['dropout']
        num_hidden_layers = config['model']['num_hidden_layers']
        
        # Calculate combined dimension: video + team
        self.combined_dim = self.video_encoder.embed_dim + self.team_embed_dim
        # Initialize output dimensions and predictors based on task form
        if self.task_form in ['coord-reg', 'generative']:
            # Coordinate regression: output [5 agents * 3 coordinates]
            self.output_dim = config['model']['num_target_agents'] * 3
            self.train_mse = MeanSquaredError()
            self.val_mse = MeanSquaredError()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            
            if self.task_form == 'generative':
                self.latent_dim = config['model']['vae']['latent_dim']
                encoder_input_dim = self.output_dim + self.combined_dim  # target locations + conditioning
                self.vae_encoder = self._build_mlp(encoder_input_dim, self.latent_dim * 2, num_hidden_layers, hidden_dim, dropout)
                decoder_input_dim = self.latent_dim + self.combined_dim  # latent + conditioning
                self.predictor = self._build_mlp(decoder_input_dim, self.output_dim, num_hidden_layers, hidden_dim, dropout)
            else:
                self.predictor = self._build_mlp(self.combined_dim, self.output_dim, num_hidden_layers, hidden_dim, dropout)
        
        elif self.task_form in ['multi-label-cls', 'multi-output-reg']:
            # Place-based classification/regression: output [num_places]
            self.output_dim = config['num_places']
            self.predictor = self._build_mlp(self.combined_dim, self.output_dim, num_hidden_layers, hidden_dim, dropout)
        
        elif self.task_form in ['grid-cls', 'density-cls']:
            # Grid-based classification: output [grid_resolution^2]
            grid_resolution = config['data'].get('grid_resolution', 10)
            self.output_dim = grid_resolution * grid_resolution
            self.predictor = self._build_mlp(self.combined_dim, self.output_dim, num_hidden_layers, hidden_dim, dropout)
        
        # Initialize loss functions for regression mode
        self._init_loss_functions()
        self.coordinate_scaler = None
        self.test_predictions = []
        self.test_targets = []
        self.output_dir = config['path']['exp']
        
    def init_video_encoder(self, config):
        return VideoEncoderBaseline(config)
    
    def _build_mlp(self, input_dim, output_dim, num_hidden_layers, hidden_dim, dropout):
        """Build a multi-layer perceptron with specified number of hidden layers."""
        layers = []
        current_dim = input_dim
        
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _init_loss_functions(self):
        """Initialize loss functions based on task form."""
        if self.task_form in ['coord-reg', 'generative']:
            # Geometric loss functions for coordinate regression
            valid_loss_fns = ['mse', 'sinkhorn', 'hausdorff', 'energy']
            if self.loss_fn not in valid_loss_fns:
                raise ValueError(f"Invalid loss_fn '{self.loss_fn}'. Must be one of {valid_loss_fns}")
            
            if self.loss_fn == 'sinkhorn':
                self.geometric_loss = SamplesLoss(
                    loss="sinkhorn", 
                    p=2, 
                    blur=self.sinkhorn_blur, 
                    scaling=self.sinkhorn_scaling
                )
            elif self.loss_fn == 'hausdorff':
                self.geometric_loss = SamplesLoss(loss="hausdorff", p=2)
            elif self.loss_fn == 'energy':
                self.geometric_loss = SamplesLoss(loss="energy", p=2)
                print(f"Initialized {self.loss_fn} loss for coordinate regression mode")
    
    def set_coordinate_scaler(self, scaler):
        """Set the coordinate scaler for coordinate-based tasks."""
        if self.task_form in ['coord-reg', 'generative']:
            self.coordinate_scaler = scaler
            if scaler is not None:
                print(f"Set coordinate scaler with data_min_: {scaler.data_min_}, data_max_: {scaler.data_max_}, scale_: {scaler.scale_}")
            else:
                print("Coordinate scaler set to None")
    
    def unscale_coordinates(self, scaled_coords):
        """
        Unscale coordinates back to original coordinate space.
        Works for both predictions and targets.
        """
        if self.task_form in ['coord-reg', 'generative']:
            assert self.coordinate_scaler is not None, "Coordinate scaler should be set for coordinate regression modes"
        
        # Convert to numpy
        if torch.is_tensor(scaled_coords):
            coords_np = scaled_coords.detach().cpu().float().numpy()
            device = scaled_coords.device
            was_tensor = True
        else:
            coords_np = scaled_coords
            device = None
            was_tensor = False
        
        original_shape = coords_np.shape
        coords_flat = coords_np.reshape(-1, 3)
        unscaled_flat = self.coordinate_scaler.inverse_transform(coords_flat)
        unscaled_np = unscaled_flat.reshape(original_shape)
        
        if was_tensor:
            return torch.tensor(unscaled_np, dtype=torch.float32, device=device)
        else:
            return unscaled_np
    
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
            target_locations = target_locations.view(target_locations.shape[0], -1)  # [B, 15]
        
        # Concatenate target locations with conditioning features
        encoder_input = torch.cat([target_locations, combined_features], dim=1)  # [B, 15 + combined_dim]
        
        # Get mu and logvar from encoder
        encoder_output = self.vae_encoder(encoder_input)  # [B, latent_dim * 2]
        mu, logvar = torch.chunk(encoder_output, 2, dim=1)  # Each [B, latent_dim]
        
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
        
        # Concatenate latent with conditioning features
        decoder_input = torch.cat([z, combined_features], dim=1)  # [B, latent_dim + combined_dim]
        
        # Decode to predictions
        predictions = self.predictor(decoder_input)  # [B, 15]
        predictions = predictions.view(-1, 5, 3)  # [B, 5, 3]
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
            sample = self.decode(z, combined_features)  # [B, 5, 3]
            samples.append(sample)
        
        samples = torch.stack(samples, dim=1)  # [B, num_samples, 5, 3]
        return samples

    def process_multi_agent_video(self, video):
        """
        Process multi-agent video input.
        """
        B, A, T, C, H, W = video.shape
        
        video_reshaped = video.view(B * A, T, C, H, W)
        
        if self.agent_fusion_method == 'tublet_attentive_pooler':
            agent_embeddings = self.video_encoder(video_reshaped, return_last_hidden_state=True)  # [B*A, seq_len, embed_dim]
            num_tublets = agent_embeddings.shape[1]
            cross_agent_embeddings = agent_embeddings.view(B, A*num_tublets, -1)  # [B, A, seq_len, embed_dim]
            fused_embeddings = self.video_encoder.video_encoder.pooler(cross_agent_embeddings)  # [B, embed_dim]
        else:
            agent_embeddings = self.video_encoder(video_reshaped)  # [B*A, embed_dim]
            agent_embeddings = agent_embeddings.view(B, A, -1) # [B, A, embed_dim]
            
            if self.agent_fusion_method == 'mean':
                fused_embeddings = torch.mean(agent_embeddings, dim=1)  # [B, embed_dim]
            elif self.agent_fusion_method == 'max':
                fused_embeddings, _ = torch.max(agent_embeddings, dim=1)  # [B, embed_dim]
            elif self.agent_fusion_method == 'attention':
                fused_embeddings, _ = self.agent_attention(
                    agent_embeddings, agent_embeddings, agent_embeddings
                )  # [B, A, embed_dim]
                fused_embeddings = torch.mean(fused_embeddings, dim=1)  # [B, embed_dim]
            elif self.agent_fusion_method == 'concat':
                fused_embeddings = agent_embeddings.view(B, -1)  # [B, A*embed_dim]
                fused_embeddings = self.agent_fusion_proj(fused_embeddings)  # [B, embed_dim]
            else:
                raise ValueError(f"Unknown agent fusion method: {self.agent_fusion_method}")
        
        return fused_embeddings, agent_embeddings
    
    def compute_regression_loss(self, predictions, targets):
        """
        Compute regression loss using the configured loss function.
        """
        if self.loss_fn == 'mse':
            return F.mse_loss(predictions, targets)
        
        elif self.loss_fn in ['sinkhorn', 'hausdorff', 'energy'] and hasattr(self, 'geometric_loss'):
            batch_size = predictions.shape[0]
            predictions = predictions.float()
            targets = targets.float()
            
            total_loss = 0.0
            for i in range(batch_size):
                pred_i = predictions[i]  # [5, 3]
                target_i = targets[i]    # [5, 3]
                loss_i = self.geometric_loss(pred_i, target_i)
                total_loss += loss_i
            return total_loss / batch_size
    
    def compute_vae_loss(self, predictions, targets, mu, logvar, kl_weight):
        """
        Compute VAE loss: reconstruction loss + KL divergence.
        
        Args:
            predictions: [B, 5, 3] - reconstructed targets
            targets: [B, 5, 3] - ground truth targets
            mu: [B, latent_dim] - latent mean
            logvar: [B, latent_dim] - latent log variance
            kl_weight: float - weight for KL divergence term
            
        Returns:
            total_loss: VAE loss
            recon_loss: reconstruction loss component
            kl_loss: KL divergence component
        """
        # Reconstruction loss (same as regression loss)
        recon_loss = self.compute_regression_loss(predictions, targets)
        
        # KL divergence: D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        # Total VAE loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    

    def forward(self, batch, mode='full'):
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch containing video, team_side_encoded, and optionally enemy_locations and text
            mode: 'full' for full VAE forward pass, 'sampling' for sampling from prior
        """
        video = batch['video']  # [B, A, T, C, H, W], e.g. [32, 3, 20, 3, 224, 224]
        team_side_encoded = batch['team_side_encoded']  # [B] with values 0 (T) or 1 (CT)
        
        if len(video.shape) != 6:
            raise ValueError(f"Expected video input shape [B, A, T, C, H, W], got {video.shape}")
        
        fused_embeddings, agent_embeddings = self.process_multi_agent_video(video)
        team_embeddings = self.team_encoder(team_side_encoded)  # [B, team_embed_dim]
        
        # Combine video and team features
        combined_features = torch.cat([fused_embeddings, team_embeddings], dim=1)  # [B, combined_dim]
        
        if self.task_form == 'generative':
            if mode == 'sampling':
                # Generative mode: sample from prior (no encoder)
                predictions = self.sample_from_prior(combined_features, num_samples=1)  # [B, 1, 5, 3]
                predictions = predictions.squeeze(1)  # [B, 5, 3]
                
                result = {
                    'predictions': predictions,
                    'fused_embeddings': fused_embeddings,
                    'team_embeddings': team_embeddings,
                    'combined_features': combined_features,
                    'agent_embeddings': agent_embeddings,
                    'mu': None,
                    'logvar': None,
                    'z': None
                }
                
                return result
            else:
                # Full VAE mode: encode targets, reparameterize, decode
                if 'enemy_locations' not in batch:
                    raise ValueError("enemy_locations required for full VAE forward pass in generative mode")
                target_locations = batch['enemy_locations']  # [B, 5, 3]
                
                mu, logvar = self.encode(target_locations, combined_features)
                z = self.reparameterize(mu, logvar)
                predictions = self.decode(z, combined_features)
                
                result = {
                    'predictions': predictions,
                    'fused_embeddings': fused_embeddings,
                    'team_embeddings': team_embeddings,
                    'combined_features': combined_features,
                    'agent_embeddings': agent_embeddings,
                    'mu': mu,
                    'logvar': logvar,
                    'z': z
                }
                
                return result
        else:
            # Standard forward pass for regression/classification
            predictions = self.predictor(combined_features)
            
            if self.task_form == 'regression':
                predictions = predictions.view(-1, 5, 3)

            result = {
                'predictions': predictions,
                'fused_embeddings': fused_embeddings,
                'team_embeddings': team_embeddings,
                'combined_features': combined_features,
                'agent_embeddings': agent_embeddings  # [B, A, embed_dim]
            }
            
            return result

    @torch._dynamo.disable
    def safe_log(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        batch_size = batch["video"].shape[0]
        targets = batch["enemy_locations"]
        
        outputs = self.forward(batch)
        predictions = outputs['predictions']

        if self.task_form == 'coord-reg':
            loss = self.compute_regression_loss(predictions, targets)
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # Binary cross-entropy for multi-label classification
            loss = F.binary_cross_entropy_with_logits(predictions, targets)
        elif self.task_form == 'multi-output-reg':
            # MSE loss for count regression
            loss = F.mse_loss(predictions, targets)
        elif self.task_form == 'density-cls':
            # KL divergence or MSE for density distribution
            loss = F.mse_loss(predictions, targets)
        elif self.task_form == 'generative':
            # VAE loss with reconstruction and KL divergence
            mu, logvar = outputs['mu'], outputs['logvar']
            kl_weight = self.config['model']['vae']['kl_weight']  # kl_weight-VAE parameter
            loss, recon_loss, kl_loss = self.compute_vae_loss(predictions, targets, mu, logvar, kl_weight)
            self.safe_log('train/recon_loss', recon_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
            self.safe_log('train/kl_loss', kl_loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False)
            
        self.safe_log('train/loss', loss, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate MSE/MAE for coordinate regression modes
        if self.task_form in ['coord-reg', 'generative']:
            self.train_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.train_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))

            self.safe_log('train/mse', self.train_mse, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('train/mae', self.train_mae, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @torch._dynamo.disable
    def _tensor_only_batch(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v
        return out

    def validation_step(self, batch, batch_idx):
        batch = self._tensor_only_batch(batch)
        batch_size = batch["video"].shape[0]
        targets = batch["enemy_locations"]
        
        # Full VAE forward pass
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        if self.task_form == 'coord-reg':
            loss = self.compute_regression_loss(predictions, targets)
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            loss = F.binary_cross_entropy_with_logits(predictions, targets)
        elif self.task_form == 'multi-output-reg':
            loss = F.mse_loss(predictions, targets)
        elif self.task_form == 'density-cls':
            loss = F.mse_loss(predictions, targets)
        elif self.task_form == 'generative':
            # VAE loss with reconstruction and KL divergence
            mu, logvar = outputs['mu'], outputs['logvar']
            kl_weight = self.config['model']['vae']['kl_weight']
            loss, recon_loss, kl_loss = self.compute_vae_loss(predictions, targets, mu, logvar, kl_weight)
            
            # Log VAE-specific metrics
            self.safe_log('val/recon_loss', recon_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            self.safe_log('val/kl_loss', kl_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            
            # Also test generative sampling
            gen_outputs = self.forward(batch, mode='generative')
            gen_predictions = gen_outputs['predictions']
            gen_loss = self.compute_regression_loss(gen_predictions, targets)
            self.safe_log('val/gen_loss', gen_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            
        self.safe_log('val/loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.task_form in ['coord-reg', 'generative']:
            self.val_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.val_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.safe_log('val/mse', self.val_mse, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('val/mae', self.val_mae, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def on_test_start(self):
        """Clear test statistics at the start of testing."""
        self.test_team_sides = []  # Track team sides for each sample
        self.test_targets = []
        self.test_predictions = []
        self.test_targets_unscaled = []
        self.test_predictions_unscaled = []
        # Store raw test samples for multiple predictions
        self.test_raw_samples = []
        self.test_raw_samples_by_team = {'T': [], 'CT': []}

    def test_step(self, batch, batch_idx):
        """Test step - similar to validation step but with test metrics."""
        # Store team sides before converting to tensor-only batch
        team_sides = batch['team_side']
        
        samples_per_team = 5
        for i in range(batch['video'].shape[0]):
            team_side = team_sides[i]
            
            # Check if we need more samples from this team
            if len(self.test_raw_samples_by_team[team_side]) < samples_per_team:
                sample = {
                    'video': batch['video'][i:i+1].clone(),  # Keep batch dimension
                    'team_side_encoded': batch['team_side_encoded'][i:i+1].clone(),
                    'enemy_locations': batch['enemy_locations'][i:i+1].clone(),
                    'team_side': team_side
                }
                self.test_raw_samples_by_team[team_side].append(sample)
                
            # Break early if we have enough samples from both teams
            if (len(self.test_raw_samples_by_team['T']) >= samples_per_team and 
                len(self.test_raw_samples_by_team['CT']) >= samples_per_team):
                break
        
        batch = self._tensor_only_batch(batch)
        batch_size = batch["video"].shape[0]
        targets = batch["enemy_locations"]
        
        # Full VAE forward pass (or standard forward for other modes)
        outputs = self.forward(batch, mode='full')
        predictions = outputs['predictions']
        
        if self.task_form == 'coord-reg':
            loss = self.compute_regression_loss(predictions, targets)
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            loss = F.binary_cross_entropy_with_logits(predictions, targets)
        elif self.task_form == 'multi-output-reg':
            loss = F.mse_loss(predictions, targets)
        elif self.task_form == 'density-cls':
            loss = F.mse_loss(predictions, targets)
        elif self.task_form == 'generative':
            # VAE loss with reconstruction and KL divergence
            mu, logvar = outputs['mu'], outputs['logvar']
            kl_weight = self.config['model']['vae']['kl_weight']
            loss, recon_loss, kl_loss = self.compute_vae_loss(predictions, targets, mu, logvar, kl_weight)
            
            # Log VAE-specific metrics
            self.safe_log('test/recon_loss', recon_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            self.safe_log('test/kl_loss', kl_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            
            # Also test generative sampling
            gen_outputs = self.forward(batch, mode='generative')
            gen_predictions = gen_outputs['predictions']
            gen_loss = self.compute_regression_loss(gen_predictions, targets)
            self.safe_log('test/gen_loss', gen_loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False)
            
        self.safe_log('test/loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        
        # Initialize and calculate MSE/MAE for coordinate regression modes
        if self.task_form in ['coord-reg', 'generative']:
            if not hasattr(self, 'test_mse'):
                self.test_mse = MeanSquaredError().to(self.device)
                self.test_mae = MeanAbsoluteError().to(self.device)
            
            self.test_mse(predictions.view(batch_size, -1), targets.view(batch_size, -1))
            self.test_mae(predictions.view(batch_size, -1), targets.view(batch_size, -1))
        
        self.test_team_sides.extend(team_sides)
        
        if self.task_form in ['coord-reg', 'generative']:
            predictions_unscaled = self.unscale_coordinates(predictions)
            targets_unscaled = self.unscale_coordinates(targets)
            
            self.test_predictions.append(predictions.cpu().float())
            self.test_targets.append(targets.cpu().float())

            self.test_predictions_unscaled.extend(predictions_unscaled.cpu().float().numpy())
            self.test_targets_unscaled.extend(targets_unscaled.cpu().float().numpy())
            self.safe_log('test/mse', self.test_mse, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
            self.safe_log('test/mae', self.test_mae, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Store predictions and targets for classification/grid-based analysis
            self.test_predictions.append(predictions.cpu().float())
            self.test_targets.append(targets.cpu().float())
            
        return loss 

    def on_test_epoch_end(self):
        """Calculate custom metrics and create plots at the end of test epoch."""
        # Properly concatenate tensors from test predictions and targets
        predictions = torch.cat(self.test_predictions, dim=0).cpu().numpy()
        targets = torch.cat(self.test_targets, dim=0).cpu().numpy()
        team_sides = np.array(self.test_team_sides)
        unique_teams, team_counts = np.unique(team_sides, return_counts=True)
        
        output_dir = Path(self.output_dir)
        if hasattr(self, 'checkpoint_name'):
            plots_dir = output_dir / "test_analysis" / f"enemy_location_{self.checkpoint_name}"
        else:
            plots_dir = output_dir / "test_analysis" / "enemy_location"
        
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving enemy location prediction test analysis to: {plots_dir}")
        
        test_results = self._calculate_custom_metrics(predictions, targets)
        team_specific_metrics = self._calculate_team_specific_metrics(predictions, targets, team_sides)

        test_results['team_specific_metrics'] = team_specific_metrics
        
        for team in ['CT', 'T']:
            if team in team_specific_metrics:
                metrics = team_specific_metrics[team]
                if self.task_form in ['coord-reg', 'generative']:
                    geom_metrics = metrics['geometric_distances']
                    self.safe_log(f'test/{team.lower()}_mse', metrics['mse'], on_epoch=True)
                    self.safe_log(f'test/{team.lower()}_mae', metrics['mae'], on_epoch=True)
                    self.safe_log(f'test/{team.lower()}_chamfer_distance', geom_metrics['chamfer_distance_mean'], on_epoch=True)
                    self.safe_log(f'test/{team.lower()}_wasserstein_distance', geom_metrics['wasserstein_distance_mean'], on_epoch=True)
                elif self.task_form in ['multi-label-cls', 'multi-output-reg', 'grid-cls', 'density-cls']:
                    # Classification/grid-based metrics
                    if 'exact_accuracy' in metrics:
                        self.safe_log(f'test/{team.lower()}_exact_accuracy', metrics['exact_accuracy'], on_epoch=True)
                    if 'l1_count_error' in metrics:
                        self.safe_log(f'test/{team.lower()}_l1_count_error', metrics['l1_count_error'], on_epoch=True)
                    if 'kl_divergence' in metrics:
                        self.safe_log(f'test/{team.lower()}_kl_divergence', metrics['kl_divergence'], on_epoch=True)
        
        if self.task_form in ['coord-reg', 'generative']:
            geom_metrics = test_results['geometric_distances']
            self.safe_log('test/chamfer_distance', geom_metrics['chamfer_distance_mean'], on_epoch=True)
            self.safe_log('test/wasserstein_distance', geom_metrics['wasserstein_distance_mean'], on_epoch=True)
        elif self.task_form in ['multi-label-cls', 'multi-output-reg', 'grid-cls', 'density-cls']:
            # Classification/grid-based metrics
            if 'exact_accuracy' in test_results:
                self.safe_log('test/exact_accuracy', test_results['exact_accuracy'], on_epoch=True)
            if 'l1_count_error' in test_results:
                self.safe_log('test/l1_count_error', test_results['l1_count_error'], on_epoch=True)
            if 'kl_divergence' in test_results:
                self.safe_log('test/kl_divergence', test_results['kl_divergence'], on_epoch=True)
            if 'multinomial_loss' in test_results:
                self.safe_log('test/multinomial_loss', test_results['multinomial_loss'], on_epoch=True)

        test_results['num_agents'] = self.config['data']['num_agents']
        test_results['agent_fusion_method'] = self.agent_fusion_method
        test_results['task_form'] = self.task_form
        test_results['team_distribution'] = dict(zip(unique_teams.tolist(), team_counts.tolist()))
        if self.task_form in ['coord-reg', 'generative']:
            test_results['loss_function'] = self.loss_fn
            test_results['coordinate_scaling'] = self.coordinate_scaler is not None
            if self.coordinate_scaler is not None:
                test_results['scaler_data_min'] = self.coordinate_scaler.data_min_.tolist()
                test_results['scaler_data_max'] = self.coordinate_scaler.data_max_.tolist()
                test_results['scaler_scale'] = self.coordinate_scaler.scale_.tolist()
            if self.loss_fn == 'sinkhorn':
                test_results['sinkhorn_blur'] = self.sinkhorn_blur
                test_results['sinkhorn_scaling'] = self.sinkhorn_scaling
            if self.task_form == 'generative':
                test_results['latent_dim'] = self.latent_dim
                test_results['kl_weight'] = self.config['model']['vae']['kl_weight']
        
        results_file = plots_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=json_serializable)
        print(f"Enemy location prediction test results saved to: {results_file}")
        create_prediction_plots(self.task_form, predictions, targets, plots_dir, team_sides)
        
        # Create KDE heatmaps for selected test samples (for regression and generative modes)
        # Combine samples from both teams (5 from each)
        combined_samples = []
        for team in ['T', 'CT']:
            combined_samples.extend(self.test_raw_samples_by_team[team])
        
        if self.task_form in ['regression', 'generative'] and len(combined_samples) > 0:
            print(f"Creating KDE heatmaps for {len(combined_samples)} test samples...")
            print(f"  T samples: {len(self.test_raw_samples_by_team['T'])}")
            print(f"  CT samples: {len(self.test_raw_samples_by_team['CT'])}")
            
            predictions_list = []
            targets_list = []
            team_sides_list = []
            scaled_predictions_list = []
            scaled_targets_list = []
            
            for sample in combined_samples:
                multi_predictions, target = self.generate_multiple_predictions(sample, num_predictions=100)
                
                # Get scaled versions for Chamfer distance calculation
                # Need to scale the first prediction back to scaled coordinates
                first_pred_unscaled = torch.tensor(multi_predictions[0:1], dtype=torch.float32)  # [1, 5, 3]
                
                # Scale it back using inverse of the coordinate scaler
                if self.coordinate_scaler is not None:
                    # Convert unscaled back to scaled for CD calculation
                    first_pred_flat = first_pred_unscaled.view(-1, 3).numpy()  # [5, 3]
                    first_pred_scaled = self.coordinate_scaler.transform(first_pred_flat)  # [5, 3]
                    scaled_pred = torch.tensor(first_pred_scaled.reshape(1, 5, 3), dtype=torch.float32)  # [1, 5, 3]
                else:
                    scaled_pred = first_pred_unscaled
                
                scaled_target = sample['enemy_locations']  # Already scaled [1, 5, 3]
                
                # Ensure both tensors are on the same device (move to device of the model)
                device = next(self.parameters()).device
                scaled_pred = scaled_pred.to(device)
                scaled_target = scaled_target.to(device)
                
                predictions_list.append(multi_predictions)
                targets_list.append(target)
                team_sides_list.append(sample['team_side'])
                scaled_predictions_list.append(scaled_pred)
                scaled_targets_list.append(scaled_target)
                            
            create_prediction_heatmaps_grid(predictions_list, targets_list, team_sides_list, 
                                          scaled_predictions_list, scaled_targets_list, 
                                          plots_dir, map_name="de_mirage")

    def configure_optimizers(self):
        opt_config = self.config['optimization']
        lr = opt_config['lr']
        weight_decay = opt_config['weight_decay']
        fused_optimizer = opt_config['fused_optimizer']
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            fused=fused_optimizer,
        )
        
        return {
            'optimizer': optimizer,
        }
    
    
    
    def _calculate_geometric_distances(self):
        """Calculate Chamfer and Wasserstein distances for regression and generative modes using scaled coordinates."""
        if self.task_form not in ['regression', 'generative'] or not hasattr(self, 'test_predictions') or len(self.test_predictions) == 0:
            assert False, "Geometric distances cannot be calculated for non-regression/generative mode or if test_predictions is not set"
        
        all_predictions = torch.cat(self.test_predictions, dim=0).float()  # [N, 5, 3] - scaled coordinates
        all_targets = torch.cat(self.test_targets, dim=0).float()  # [N, 5, 3] - scaled coordinates
        
        chamfer_distances = chamfer_distance_batch(all_predictions, all_targets).cpu().numpy()
        
        wasserstein_distances = []
        for i in range(all_predictions.shape[0]):
            pred_i = all_predictions[i].cpu().numpy()  # [5, 3]
            target_i = all_targets[i].cpu().numpy()    # [5, 3]
            
            # Flatten the coordinates for 1D Wasserstein distance
            pred_flat = pred_i.flatten()
            target_flat = target_i.flatten()
            
            # Calculate 1D Wasserstein distance
            wd = wasserstein_distance(pred_flat, target_flat)
            wasserstein_distances.append(wd)
        
        wasserstein_distances = np.array(wasserstein_distances)
        
        geometric_metrics = {
            'chamfer_distance_mean': float(np.mean(chamfer_distances)),
            'chamfer_distance_std': float(np.std(chamfer_distances)),
            'wasserstein_distance_mean': float(np.mean(wasserstein_distances)),
            'wasserstein_distance_std': float(np.std(wasserstein_distances)),
            'num_valid_samples': len(chamfer_distances)
        }
        return geometric_metrics
            
    def _calculate_custom_metrics(self, predictions, targets):
        """Calculate comprehensive test metrics."""
        per_dim_metrics = {}
        
        if self.task_form in ['coord-reg', 'generative']:
            # For coordinate regression and generative: calculate MSE/MAE
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # predictions and targets shape: [N, 5, 3]
            predictions_flat = predictions.reshape(-1, 15)  # [N, 15]
            targets_flat = targets.reshape(-1, 15)
            
            for i in range(5):  # 5 players
                for j, coord in enumerate(['X', 'Y', 'Z']):
                    dim_idx = i * 3 + j
                    pred_dim = predictions_flat[:, dim_idx]
                    target_dim = targets_flat[:, dim_idx]
                    
                    per_dim_metrics[f'player_{i}_{coord}_mse'] = np.mean((pred_dim - target_dim) ** 2)
                    per_dim_metrics[f'player_{i}_{coord}_mae'] = np.mean(np.abs(pred_dim - target_dim))
            
            # Add geometric distances for regression
            geometric_metrics = self._calculate_geometric_distances()
            
            return {
                'overall_mse': float(mse),
                'overall_mae': float(mae),
                'per_dimension_metrics': per_dim_metrics,
                'geometric_distances': geometric_metrics,
                'num_samples': len(targets),
                'predictions_shape': list(predictions.shape),
                'targets_shape': list(targets.shape)
            }
        
        elif self.task_form in ['multi-label-cls', 'grid-cls']:
            # For binary classification: predictions are logits [N, num_outputs], targets are binary [N, num_outputs]
            predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
            pred_probs = torch.sigmoid(predictions_tensor).numpy()
            pred_binary = (pred_probs > 0.5).astype(float)
            
            # Calculate binary classification metrics
            accuracy = np.mean(pred_binary == targets)
            precision = np.sum((pred_binary == 1) & (targets == 1)) / (np.sum(pred_binary == 1) + 1e-8)
            recall = np.sum((pred_binary == 1) & (targets == 1)) / (np.sum(targets == 1) + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Calculate per-cell/place accuracy
            for i in range(predictions.shape[1]):
                per_dim_metrics[f'cell_{i}_accuracy'] = np.mean(pred_binary[:, i] == targets[:, i])
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'per_dimension_metrics': per_dim_metrics,
                'num_samples': len(targets),
                'predictions_shape': list(predictions.shape),
                'targets_shape': list(targets.shape)
            }
        
        elif self.task_form == 'multi-output-reg':
            # For count regression: predictions are counts [N, num_places], targets are counts [N, num_places]
            pred_counts = predictions
            
            # Calculate count regression metrics
            exact_accuracy = exact_match_accuracy(pred_counts, targets)
            l1_error = l1_count_error(pred_counts, targets)
            # For count regression, convert to distribution for KL
            pred_probs = pred_counts / (pred_counts.sum(axis=1, keepdims=True) + 1e-8)
            kl_div = kl_divergence_histogram(pred_probs, targets, n_agents=5)
            
            # Calculate per-place metrics
            for i in range(predictions.shape[1]):  # num_places
                pred_place_counts = pred_counts[:, i]
                target_place_counts = targets[:, i]
                
                # Accuracy for this place (how often we get the exact count right)
                place_exact_matches = np.round(pred_place_counts) == target_place_counts
                per_dim_metrics[f'place_{i}_accuracy'] = np.mean(place_exact_matches)
                
                # Average predicted vs actual counts
                per_dim_metrics[f'place_{i}_pred_count_mean'] = np.mean(pred_place_counts)
                per_dim_metrics[f'place_{i}_actual_count_mean'] = np.mean(target_place_counts)
            
            # Calculate MSE loss for count regression
            mse_loss = np.mean((pred_counts - targets) ** 2)
            
            return {
                'mse_loss': float(mse_loss),
                'exact_accuracy': float(exact_accuracy),
                'l1_count_error': float(l1_error),
                'kl_divergence': float(kl_div),
                'per_dimension_metrics': per_dim_metrics,
                'num_samples': len(targets),
                'predictions_shape': list(predictions.shape),
                'targets_shape': list(targets.shape)
            }
        
        elif self.task_form == 'density-cls':
            # For density distribution: predictions and targets are distributions [N, num_cells]
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # KL divergence for density distributions
            pred_probs = predictions / (predictions.sum(axis=1, keepdims=True) + 1e-8)
            target_probs = targets / (targets.sum(axis=1, keepdims=True) + 1e-8)
            kl_div = np.mean(np.sum(target_probs * np.log((target_probs + 1e-8) / (pred_probs + 1e-8)), axis=1))
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'kl_divergence': float(kl_div),
                'per_dimension_metrics': per_dim_metrics,
                'num_samples': len(targets),
                'predictions_shape': list(predictions.shape),
                'targets_shape': list(targets.shape)
            }
    
    def _calculate_team_specific_metrics(self, predictions, targets, team_sides):
        """Calculate metrics separately for each team."""
        team_metrics = {}
        
        for team in ['CT', 'T']:
            team_mask = team_sides == team
            if not np.any(team_mask):
                continue
                
            team_predictions = predictions[team_mask]
            team_targets = targets[team_mask]
            
            if len(team_predictions) == 0:
                continue
            
            # Calculate appropriate metrics based on task form
            if self.task_form in ['coord-reg', 'generative']:
                mse = np.mean((team_predictions - team_targets) ** 2)
                mae = np.mean(np.abs(team_predictions - team_targets))
                
                team_metrics[team] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'num_samples': len(team_predictions)
                }
            elif self.task_form in ['multi-label-cls', 'grid-cls']:
                # Binary classification metrics
                team_pred_tensor = torch.tensor(team_predictions, dtype=torch.float32)
                team_pred_probs = torch.sigmoid(team_pred_tensor).numpy()
                team_pred_binary = (team_pred_probs > 0.5).astype(float)
                
                accuracy = np.mean(team_pred_binary == team_targets)
                precision = np.sum((team_pred_binary == 1) & (team_targets == 1)) / (np.sum(team_pred_binary == 1) + 1e-8)
                recall = np.sum((team_pred_binary == 1) & (team_targets == 1)) / (np.sum(team_targets == 1) + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                team_metrics[team] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'num_samples': len(team_predictions)
                }
            elif self.task_form == 'multi-output-reg':
                # Count regression metrics
                team_pred_counts = team_predictions
                exact_accuracy = exact_match_accuracy(team_pred_counts, team_targets)
                l1_error = l1_count_error(team_pred_counts, team_targets)
                team_pred_probs = team_pred_counts / (team_pred_counts.sum(axis=1, keepdims=True) + 1e-8)
                kl_div = kl_divergence_histogram(team_pred_probs, team_targets, n_agents=5)
                
                team_metrics[team] = {
                    'exact_accuracy': float(exact_accuracy),
                    'l1_count_error': float(l1_error),
                    'kl_divergence': float(kl_div),
                    'num_samples': len(team_predictions)
                }
            elif self.task_form == 'density-cls':
                # Density distribution metrics
                mse = np.mean((team_predictions - team_targets) ** 2)
                mae = np.mean(np.abs(team_predictions - team_targets))
                
                team_metrics[team] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'num_samples': len(team_predictions)
                }
            
            if self.task_form in ['coord-reg', 'generative']:
                team_indices = np.where(team_mask)[0]
                
                team_pred_tensors = []
                team_target_tensors = []
                
                current_idx = 0
                for tensor_batch in self.test_predictions:
                    batch_size = tensor_batch.shape[0]
                    batch_indices = np.arange(current_idx, current_idx + batch_size)
                    
                    team_batch_mask = np.isin(batch_indices, team_indices)
                    
                    if np.any(team_batch_mask):
                        team_pred_tensors.append(tensor_batch[team_batch_mask])
                    
                    current_idx += batch_size
                
                current_idx = 0
                for tensor_batch in self.test_targets:
                    batch_size = tensor_batch.shape[0]
                    batch_indices = np.arange(current_idx, current_idx + batch_size)
                    
                    team_batch_mask = np.isin(batch_indices, team_indices)
                    
                    if np.any(team_batch_mask):
                        team_target_tensors.append(tensor_batch[team_batch_mask])
                    
                    current_idx += batch_size
                
                if team_pred_tensors and team_target_tensors:
                    team_pred_combined = torch.cat(team_pred_tensors, dim=0)
                    team_target_combined = torch.cat(team_target_tensors, dim=0)
                    
                    chamfer_distances = chamfer_distance_batch(team_pred_combined, team_target_combined).cpu().numpy()
                    
                    wasserstein_distances = []
                    for i in range(team_pred_combined.shape[0]):
                        pred_i = team_pred_combined[i].cpu().numpy().flatten()
                        target_i = team_target_combined[i].cpu().numpy().flatten()
                        wd = wasserstein_distance(pred_i, target_i)
                        wasserstein_distances.append(wd)
                    
                    wasserstein_distances = np.array(wasserstein_distances)
                    
                    team_metrics[team]['geometric_distances'] = {
                        'chamfer_distance_mean': float(np.mean(chamfer_distances)),
                        'chamfer_distance_std': float(np.std(chamfer_distances)),
                        'wasserstein_distance_mean': float(np.mean(wasserstein_distances)),
                        'wasserstein_distance_std': float(np.std(wasserstein_distances)),
                        'num_valid_samples': len(chamfer_distances)
                    }
        
        return team_metrics
    
    @torch.inference_mode()
    def generate_multiple_predictions(self, sample, num_predictions=100):
        """
        Generate multiple predictions for a single test sample using dropout or VAE sampling.
        
        Args:
            sample: Dictionary containing 'video', 'team_side_encoded', 'enemy_locations'
            num_predictions: Number of predictions to generate
            
        Returns:
            predictions: numpy array of shape [num_predictions, 5, 3] (unscaled coordinates)
            target: numpy array of shape [5, 3] (unscaled ground truth)
        """
        self.eval()
        
        predictions = []
        for _ in range(num_predictions):
            if self.task_form == 'generative':
                outputs = self.forward(sample, mode='sampling')
            else:
                outputs = self.forward(sample, mode='full')
            
            pred = outputs['predictions']  # [1, 5, 3] or [1, num_places]
            
            if self.task_form in ['regression', 'generative']:
                pred_unscaled = self.unscale_coordinates(pred)
                predictions.append(pred_unscaled.cpu().numpy()[0])  # Remove batch dimension
            elif self.task_form == 'classification':
                raise NotImplementedError("No support for multiple predictions generation for classification")
        
        predictions = np.array(predictions)  # [num_predictions, 5, 3]
        
        # Get unscaled ground truth
        if self.task_form in ['regression', 'generative']:
            target_unscaled = self.unscale_coordinates(sample['enemy_locations'])
            target = target_unscaled.cpu().numpy()[0]  # Remove batch dimension, shape [5, 3]
        else:
            target = sample['enemy_locations'].cpu().numpy()[0]
            
        return predictions, target
