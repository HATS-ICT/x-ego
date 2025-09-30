import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
import torch._dynamo
    
from itertools import permutations
from .encoders import CTFMVideoEncoderModel, CTFMAudioEncoderModel, CTFMTextEncoderModel


class CTFMContrastive(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Get contrastive learning configuration
        contrastive_config = config['training']['contrastive']
        self.shared_proj_dim = contrastive_config['project_to_shared_dim']
        self.contrastive_style = contrastive_config['style']
        self.use_symmetric_loss = contrastive_config['use_symmetric_loss']
        
        # Get modalities from config
        self.modalities = config['model']['modalities']
        self.has_video = 'video' in self.modalities
        self.has_audio = 'audio' in self.modalities
        self.has_text = 'text' in self.modalities
        
        # Initialize encoders conditionally based on modalities
        self.video_encoder = None
        self.audio_encoder = None
        self.text_encoder = None
        
        if self.has_video:
            video_config = config['model']['encoder']['video'].copy()
            video_config['shared_proj_dim'] = self.shared_proj_dim
            self.video_encoder = CTFMVideoEncoderModel(video_config)
            self.video_dim = self.shared_proj_dim
        
        if self.has_audio:
            audio_config = config['model']['encoder']['audio'].copy()
            audio_config['shared_proj_dim'] = self.shared_proj_dim
            self.audio_encoder = CTFMAudioEncoderModel(audio_config)
            self.audio_dim = self.shared_proj_dim
        
        if self.has_text:
            text_config = config['model']['encoder']['text'].copy()
            text_config['shared_proj_dim'] = self.shared_proj_dim
            self.text_encoder = CTFMTextEncoderModel(text_config)
            self.text_dim = self.shared_proj_dim
        
        # Initialize learnable parameters
        logit_temp_init = contrastive_config['logit_temp_init']
        logit_bias_init = contrastive_config['logit_bias_init']
        self.logit_temp = nn.Parameter(torch.tensor(logit_temp_init, dtype=torch.float32))
        self.logit_bias = nn.Parameter(torch.tensor(logit_bias_init, dtype=torch.float32))
        
        if self.contrastive_style == 'siglip':
            self.criterion = self._sigmoid_loss
        elif self.contrastive_style == 'clip':
            self.criterion = self._info_nce_loss
        else:
            raise ValueError(f"Invalid contrastive style: {self.contrastive_style}. Only 'siglip' and 'clip' are supported.")


    def forward(self, batch, return_embeddings=False):
        embeddings = self.get_embeddings(batch)
        
        vision_embeddings = embeddings.get('video_embeddings_pooled')
        audio_embeddings = embeddings.get('audio_embeddings_pooled')
        text_embeddings = embeddings.get('text_embeddings_pooled')
        
        # Normalize embeddings for available modalities
        if vision_embeddings is not None:
            vision_embeddings = vision_embeddings / vision_embeddings.norm(p=2, dim=-1, keepdim=True)
        if audio_embeddings is not None:
            audio_embeddings = audio_embeddings / audio_embeddings.norm(p=2, dim=-1, keepdim=True)
        if text_embeddings is not None:
            text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
        
        if return_embeddings:
            if vision_embeddings is not None:
                batch["video_embeddings"] = vision_embeddings
                batch["video_embeddings_normed"] = vision_embeddings
            if audio_embeddings is not None:
                batch["audio_embeddings"] = audio_embeddings
                batch["audio_embeddings_normed"] = audio_embeddings
            if text_embeddings is not None:
                batch["text_embeddings"] = text_embeddings
                batch["text_embeddings_normed"] = text_embeddings
                
        # Compute pairwise similarities for available modality pairs
        logits = {}
        device = vision_embeddings.device
        logit_scale = self.logit_temp.to(device).exp()
        logit_bias = self.logit_bias.to(device)
        
        # Video-Audio pairs (only if both modalities are available)
        if vision_embeddings is not None and audio_embeddings is not None:
            logits_per_audio = torch.matmul(audio_embeddings, vision_embeddings.t()) * logit_scale + logit_bias
            logits['audio_video'] = logits_per_audio
            logits['video_audio'] = logits_per_audio.t()
            
        # Audio-Text pairs (only if both modalities are available)
        if audio_embeddings is not None and text_embeddings is not None:
            num_text_per_clip = text_embeddings.shape[1]
            logits['text_audio'] = []
            logits['audio_text'] = []
            for i in range(num_text_per_clip):
                logits_per_text = torch.matmul(text_embeddings[:, i, :], audio_embeddings.t()) * logit_scale + logit_bias
                logits['text_audio'].append(logits_per_text)
                logits['audio_text'].append(logits_per_text.t())
            logits['text_audio'] = torch.stack(logits['text_audio'], dim=1).permute(1, 0, 2)
            logits['audio_text'] = torch.stack(logits['audio_text'], dim=1).permute(1, 0, 2)
            
        # Video-Text pairs (only if both modalities are available)
        if vision_embeddings is not None and text_embeddings is not None:
            num_text_per_clip = text_embeddings.shape[1]
            logits['text_video'] = []
            logits['video_text'] = []
            for i in range(num_text_per_clip):
                logits_per_text = torch.matmul(text_embeddings[:, i, :], vision_embeddings.t()) * logit_scale + logit_bias
                logits['text_video'].append(logits_per_text)
                logits['video_text'].append(logits_per_text.t())
            logits['text_video'] = torch.stack(logits['text_video'], dim=1).permute(1, 0, 2)
            logits['video_text'] = torch.stack(logits['video_text'], dim=1).permute(1, 0, 2)
            
        return logits, batch
    
    @torch.inference_mode()
    def get_contrastive_vision_embedding(self, video_feature):
        """Get contrastive embeddings from video, audio, and text encoders"""
        video_features_pooled = self.video_encoder(video_feature).last_hidden_state
        vision_embeddings = video_features_pooled / video_features_pooled.norm(p=2, dim=-1, keepdim=True)
        return vision_embeddings
    
    @torch.inference_mode()
    def get_contrastive_audio_embedding(self, audio_feature, attention_mask=None):
        """Get contrastive embeddings from video, audio, and text encoders"""
        audio_features_pooled = self.audio_encoder(audio_feature, attention_mask=attention_mask).last_hidden_state
        audio_embeddings = audio_features_pooled / audio_features_pooled.norm(p=2, dim=-1, keepdim=True)
        return audio_embeddings
    
    @torch.inference_mode()
    def get_contrastive_text_embedding(self, text_feature, attention_mask=None):
        """Get contrastive embeddings from video, audio, and text encoders"""
        text_features_pooled = self.text_encoder(text_feature, attention_mask=attention_mask).last_hidden_state
        text_embeddings = text_features_pooled / text_features_pooled.norm(p=2, dim=-1, keepdim=True)
        return text_embeddings

    def get_embeddings(self, batch, is_precomputed=False):
        """Get embeddings from available encoders"""
        embeddings = {}
        
        if is_precomputed:
            # For precomputed embeddings, assume they are already pooled and projected
            if self.has_video and "video_embeddings" in batch:
                embeddings['video_embeddings_pooled'] = batch["video_embeddings"]
            if self.has_audio and "audio_embeddings" in batch:
                embeddings['audio_embeddings_pooled'] = batch["audio_embeddings"]
            if self.has_text and "text_embeddings" in batch:
                embeddings['text_embeddings_pooled'] = batch["text_embeddings"]
        else:
            # Get embeddings from encoders (already pooled and projected)
            if self.has_video and "video" in batch:
                embeddings['video_embeddings_pooled'] = self.video_encoder(batch["video"]).last_hidden_state
            if self.has_audio and "audio" in batch:
                embeddings['audio_embeddings_pooled'] = self.audio_encoder(batch["audio"], attention_mask=batch["audio_attention_mask"]).last_hidden_state
            if self.has_text and "text" in batch:
                embeddings['text_embeddings_pooled'] = self.text_encoder(batch["text"], attention_mask=batch["text_attention_mask"]).last_hidden_state
        
        return embeddings
    
    @torch._dynamo.disable
    def safe_log(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def _calculate_retrieval_accuracy(self, logits, k=1):
        """Calculate top-k retrieval accuracy"""
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        
        accuracy = 0
        for logits_item in logits:
            batch_size = logits_item.shape[0]
            labels = torch.arange(batch_size, device=logits_item.device)
            
            # Get top-k predictions
            _, top_k_indices = torch.topk(logits_item, k, dim=1)
            
            # Check if correct label is in top-k
            correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
            accuracy += correct.float().mean()

        return accuracy / len(logits)

    def _calculate_embedding_stats(self, embeddings):
        """Calculate embedding statistics"""
        if embeddings is None:
            return None, None
        
        # Calculate L2 norms
        norms = torch.norm(embeddings, p=2, dim=-1)
        mean_norm = norms.mean()
        std_norm = norms.std()
        
        return mean_norm, std_norm

    @torch._dynamo.disable
    def _calculate_similarity_stats(self, logits):
        """
        Calculate rich similarity statistics on a [B, B] similarity/logit matrix.

        Returns a dict with:
        - mean, std, min, max                          # global over all entries
        - diag_mean, diag_std                          # positives (diagonal)
        - off_mean, off_std, off_p95, off_p99          # negatives (off-diagonal)
        - margin                                       # diag_mean - off_mean
        """
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)
        
        stats_list = []
        for logits_item in logits:
            B = logits_item.size(0)
            diag = torch.diagonal(logits_item)  # [B]

            # mask out diagonal to get off-diagonal values
            off_mask = ~torch.eye(B, dtype=torch.bool, device=logits_item.device)
            off_vals = logits_item.masked_select(off_mask)

            # Convert to float32 for quantile calculation if needed
            if off_vals.dtype not in [torch.float32, torch.float64]:
                off_vals_float = off_vals.float()
            else:
                off_vals_float = off_vals

            # Stats for this item
            item_stats = {
                "mean": logits_item.mean(),
                "std": logits_item.std(),
                "diag_mean": diag.mean(),
                "off_mean": off_vals.mean(),
                "off_std": off_vals.std(unbiased=False),
                "off_p95": torch.quantile(off_vals_float, 0.95),
                "off_p99": torch.quantile(off_vals_float, 0.99),
            }
            item_stats["margin"] = item_stats["diag_mean"] - item_stats["off_mean"]
            stats_list.append(item_stats)
        
        # Average across all items
        out = {}
        for key in stats_list[0].keys():
            out[key] = sum(stats[key] for stats in stats_list) / len(stats_list)
        
        return out


    def _log_gradient_norms(self, prefix="train", batch_size=None):
        """Log gradient norms for monitoring training stability"""
        total_norm = 0.0
        param_count = 0
        
        # Use batch_size of 1 as default for gradient norms (they don't depend on batch size)
        if batch_size is None:
            batch_size = 1
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual component gradients
                if 'video' in name:
                    self.safe_log(f'{prefix}/grad_norm_video', param_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                elif 'audio' in name:
                    self.safe_log(f'{prefix}/grad_norm_audio', param_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                elif 'text' in name:
                    self.safe_log(f'{prefix}/grad_norm_text', param_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                elif 'logit_temp' in name:
                    self.safe_log(f'{prefix}/grad_norm_logit_temp', param_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                elif 'logit_bias' in name:
                    self.safe_log(f'{prefix}/grad_norm_logit_bias', param_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        if param_count > 0:
            total_norm = (total_norm ** 0.5)
            self.safe_log(f'{prefix}/grad_norm_total', total_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _info_nce_loss(self, logits_dict):
        """CLIP contrastive loss for multi-modal pairs"""
        losses = []
        batch_size = None
        device = None
        
        # Get batch size and device from any available logits
        for logits in logits_dict.values():
            if logits is not None:
                batch_size = logits.shape[0]
                device = logits.device
                break
                
        if batch_size is None:
            raise ValueError("No valid logits found for loss calculation")
            
        labels = torch.arange(batch_size, device=device)
        
        # Calculate loss for each modality pair
        modality_pairs = [
            ('audio_video', 'video_audio'),  # Audio-Video
            ('text_audio', 'audio_text'),    # Text-Audio  
            ('text_video', 'video_text')     # Text-Video
        ]
        
        for mod1_key, mod2_key in modality_pairs:
            if mod1_key in logits_dict and mod2_key in logits_dict:
                logits1 = logits_dict[mod1_key]
                logits2 = logits_dict[mod2_key]
                
                loss1 = nn.functional.cross_entropy(logits1, labels)
                loss2 = nn.functional.cross_entropy(logits2, labels)
                
                pair_loss = (loss1 + loss2) / 2
                losses.append(pair_loss)
        
        if not losses:
            raise ValueError("No valid modality pairs found for loss calculation")
            
        # Average losses from all available pairs
        total_loss = sum(losses) / len(losses)
        return total_loss
    
    @staticmethod
    def get_modality_pairs(modalities, use_symmetric_loss=False):
        if use_symmetric_loss:
            return [f"{m1}_{m2}" for m1, m2 in permutations(modalities, 2)]
        return [f"{m1}_{m2}" for m1, m2 in permutations(modalities, 2) if m1 < m2]

    def _sigmoid_loss(self, logits_dict):
        """SigLIP contrastive loss for multi-modal pairs"""
        losses = []
        batch_size = logits_dict["video_text"].shape[1]
        device = logits_dict["video_text"].device
            
        # Create labels: positive pairs on diagonal, negative pairs elsewhere
        labels = torch.eye(batch_size, device=device) * 2 - 1
        
        # Calculate SigLIP loss for each available modality pair
        modality_pairs = self.get_modality_pairs(self.modalities, self.use_symmetric_loss)
        
        loss_dict = dict()
        weight_sum = 0
        for logits_key in logits_dict.keys():
            logits = logits_dict[logits_key]
            
            # SigLIP uses sigmoid loss
            loglik = torch.nn.functional.logsigmoid(labels * logits)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            loss_dict[logits_key] = loss
            
            # Only apply loss weight if the key exists in config
            loss_weights = self.config['training']['contrastive']['loss_weights']
            if logits_key in loss_weights:
                loss_weight = loss_weights[logits_key]
                weight_sum += loss_weight
                loss = loss * loss_weight
            else:
                # Default weight of 1.0 for missing keys
                weight_sum += 1.0
            losses.append(loss)

        if not losses:
            raise ValueError("No valid logits found for SigLIP loss calculation")
            
        # Average losses from all available pairs
        total_loss = sum(losses) / max(weight_sum, 1e-6)
        loss_dict['total'] = total_loss
        return loss_dict
        
    def training_step(self, batch, batch_idx):
        batch_size = batch["video"].shape[0]
        # TODO: Remove
        # from ctfm.utils.training_utils import debug_batch_plot
        # debug_batch_plot(batch)
        # plt.imshow(batch["video"][0][2].permute(1,2,0).detach().cpu().numpy())
        # print(batch["raw_text"])

        # Get contrastive logits and embeddings in one call
        logits_dict, batch = self.forward(batch, return_embeddings=False)
        
        # Calculate contrastive loss
        loss_dict = self.criterion(logits_dict)
        loss = loss_dict['total']
        self.safe_log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=True)
        
        # Log basic metrics with train/ prefix
        for loss_key, loss_value in loss_dict.items():
            self.safe_log(f'train/loss_{loss_key}', loss_value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.safe_log('train/logit_scale', self.logit_temp.exp(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.safe_log('train/logit_bias', self.logit_bias, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.safe_log('train/learning_rate', current_lr, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Log gradient norms before optimizer step
        # Note: batch_size will default to 1 in _log_gradient_norms since gradient norms don't depend on batch size
        self._log_gradient_norms(prefix="train")


    @torch._dynamo.disable
    def _tensor_only_batch(self, batch):
        # keep only tensors you actually need downstream
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v
        return out

    def validation_step(self, batch, batch_idx):
        batch = self._tensor_only_batch(batch)
        
        batch_size = batch["video"].shape[0]
        logits_dict, batch = self.forward(batch, return_embeddings=True) 
        
        loss_dict = self.criterion(logits_dict)
        loss = loss_dict['total']
        
        for loss_key, loss_value in loss_dict.items():
            self.safe_log(f'val/loss_{loss_key}', loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.safe_log('val/logit_scale', self.logit_temp.exp(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.safe_log('val/logit_bias', self.logit_bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Log embedding statistics for available modalities
        if 'video_embeddings' in batch:
            video_embs = batch['video_embeddings']
            video_mean_norm, video_std_norm = self._calculate_embedding_stats(video_embs)
            self.safe_log('val/video_emb_unnormed_norm_mean', video_mean_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        if 'audio_embeddings' in batch:
            audio_embs = batch['audio_embeddings']
            audio_mean_norm, audio_std_norm = self._calculate_embedding_stats(audio_embs)
            self.safe_log('val/audio_emb_unnormed_norm_mean', audio_mean_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        if 'text_embeddings' in batch:
            text_embs = batch['text_embeddings']
            text_mean_norm, text_std_norm = self._calculate_embedding_stats(text_embs)
            self.safe_log('val/text_emb_unnormed_norm_mean', text_mean_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Calculate and log retrieval accuracies and similarity stats        
        for logits_key in logits_dict.keys():
            logits = logits_dict[logits_key]
            metric_prefix = f"val/{logits_key}"
            
            # Retrieval metrics (Compute for both directions, since values are asymmetric)
            top1_acc = self._calculate_retrieval_accuracy(logits, k=1)
            top3_acc = self._calculate_retrieval_accuracy(logits, k=3)
            top5_acc = self._calculate_retrieval_accuracy(logits, k=5)
            self.safe_log(f"{metric_prefix}_top1_acc", top1_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.safe_log(f"{metric_prefix}_top3_acc", top3_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.safe_log(f"{metric_prefix}_top5_acc", top5_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
            
            sim_stats = self._calculate_similarity_stats(logits)
            stat_keys = ["mean", "diag_mean", "off_mean", "off_p95", "off_p99", "margin"]
            for stat_key in stat_keys:
                self.safe_log(f'{metric_prefix}_sim_{stat_key}', sim_stats[stat_key], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss


    def test_step(self, batch, batch_idx):
        """Test step for evaluation on test set"""
        batch = self._tensor_only_batch(batch)
        
        batch_size = batch["video"].shape[0]
        logits_dict, batch = self.forward(batch, return_embeddings=True) 
        
        loss_dict = self.criterion(logits_dict)
        loss = loss_dict['total']
        
        # Get checkpoint name for logging (set by training script)
        checkpoint_name = getattr(self, 'checkpoint_name', 'unknown')
        
        for loss_key, loss_value in loss_dict.items():
            self.log(f'test/{checkpoint_name}/loss_{loss_key}', loss_value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'test/{checkpoint_name}/logit_scale', self.logit_temp.exp(), on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'test/{checkpoint_name}/logit_bias', self.logit_bias, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Log embedding statistics for available modalities
        if 'video_embeddings' in batch:
            video_embs = batch['video_embeddings']
            video_mean_norm, video_std_norm = self._calculate_embedding_stats(video_embs)
            self.log(f'test/{checkpoint_name}/video_emb_unnormed_norm_mean', video_mean_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        if 'audio_embeddings' in batch:
            audio_embs = batch['audio_embeddings']
            audio_mean_norm, audio_std_norm = self._calculate_embedding_stats(audio_embs)
            self.log(f'test/{checkpoint_name}/audio_emb_unnormed_norm_mean', audio_mean_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        if 'text_embeddings' in batch:
            text_embs = batch['text_embeddings']
            text_mean_norm, text_std_norm = self._calculate_embedding_stats(text_embs)
            self.log(f'test/{checkpoint_name}/text_emb_unnormed_norm_mean', text_mean_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        
        # Calculate and log retrieval accuracies and similarity stats
        modality_pairs = [
            ('audio_video', True),
            ('video_audio', False),
            ('text_audio', True),
            ('audio_text', False),
            ('text_video', True),
            ('video_text', False)
        ]
        
        for logits_key, compute_sim_stats in modality_pairs:
            if logits_key in logits_dict:
                logits = logits_dict[logits_key]
                metric_prefix = f"test/{checkpoint_name}/{logits_key}"
                
                # Retrieval metrics (Compute for both directions, since values are asymmetric)
                top1_acc = self._calculate_retrieval_accuracy(logits, k=1)
                top3_acc = self._calculate_retrieval_accuracy(logits, k=3)
                top5_acc = self._calculate_retrieval_accuracy(logits, k=5)
                self.log(f"{metric_prefix}_top1_acc", top1_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log(f"{metric_prefix}_top3_acc", top3_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log(f"{metric_prefix}_top5_acc", top5_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                
                # Similarity statistics (Compute just one direction, since values are symmetric)
                if compute_sim_stats:
                    sim_stats = self._calculate_similarity_stats(logits)
                    stat_keys = ["mean", "diag_mean", "off_mean", "off_p95", "off_p99", "margin"]
                    for stat_key in stat_keys:
                        self.log(f'{metric_prefix}_sim_{stat_key}', sim_stats[stat_key], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        opt_config = self.config['optimization']
        lr = opt_config['lr']
        weight_decay = opt_config['weight_decay']
        fused_optimizer = opt_config['fused_optimizer']
        
        # Configure optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            fused=fused_optimizer,
        )
        
        return {
            'optimizer': optimizer,
        }