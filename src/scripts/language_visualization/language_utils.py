"""
Language visualization utilities for SigLIP2 text-image similarity analysis.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

# Import concept vocabulary from separate file


def load_siglip2_model():
    """Load the full SigLIP2 model for text-image similarity."""
    pretrained_name = "google/siglip2-base-patch16-224"
    model = AutoModel.from_pretrained(pretrained_name)
    processor = AutoProcessor.from_pretrained(pretrained_name)
    return model, processor


def get_text_embeddings(model, processor, texts: list, device: torch.device) -> torch.Tensor:
    """
    Get text embeddings from SigLIP2 model.
    
    Args:
        model: SigLIP2 model
        processor: SigLIP2 processor
        texts: List of text strings
        device: Device to run on
        
    Returns:
        Normalized text embeddings [num_texts, embed_dim]
    """
    # Prepare text inputs - SigLIP2 expects lowercase text
    texts_lower = [t.lower() for t in texts]
    text_inputs = processor(text=texts_lower, padding="max_length", max_length=64, return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_outputs = model.text_model(**text_inputs)
        # Use pooled output
        text_embeds = text_outputs.pooler_output
        # Normalize
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    
    return text_embeds


def get_image_embeddings(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Get image embeddings from SigLIP2 vision model.
    
    Args:
        model: SigLIP2 model (full model with vision_model)
        pixel_values: Image tensor [batch, channels, height, width] - already processed
        
    Returns:
        Normalized image embeddings [batch, embed_dim]
    """
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        # Mean pooling over patches (SigLIP2 has no CLS token)
        image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
        # Normalize
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    
    return image_embeds


def compute_text_image_similarity(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between text and image embeddings.
    
    Args:
        text_embeds: [num_texts, embed_dim]
        image_embeds: [num_images, embed_dim]
        
    Returns:
        Similarity matrix [num_images, num_texts]
    """
    # Both are already normalized, so dot product gives cosine similarity
    similarity = torch.matmul(image_embeds, text_embeds.t())
    return similarity


def replace_vision_encoder_weights(full_model, finetuned_vision_state_dict):
    """
    Replace the vision encoder weights in the full SigLIP2 model with finetuned weights.
    
    Args:
        full_model: Full SigLIP2 model
        finetuned_vision_state_dict: State dict from finetuned vision model
    """
    # The finetuned weights come from just the vision_model
    # We need to load them into full_model.vision_model
    full_model.vision_model.load_state_dict(finetuned_vision_state_dict)
