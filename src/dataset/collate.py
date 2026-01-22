import torch


def contrastive_collate_fn(batch):
    """
    Custom collate function for contrastive learning dataset.
    
    Handles variable number of agents per sample by padding to max agents in batch.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video features tensor [A, ...] where A is variable
            - 'num_agents': Number of valid agents in this sample
            - 'pov_team_side': String indicating team side
            - 'pov_team_side_encoded': Team side encoded as int
            - 'agent_ids': List of agent IDs
    
    Returns:
        Dictionary with batched tensors, including agent_mask for variable agents
    """
    collated = {}
    
    # Find max agents in batch
    max_agents = max(item['video'].shape[0] for item in batch)
    
    # Pad videos to max_agents
    padded_videos = []
    agent_masks = []
    
    for item in batch:
        video = item['video']
        num_agents = video.shape[0]
        
        if num_agents < max_agents:
            # Pad with zeros: [A, T, C, H, W] -> [max_agents, T, C, H, W]
            pad_shape = (max_agents - num_agents,) + video.shape[1:]
            padding = torch.zeros(pad_shape, dtype=video.dtype)
            video = torch.cat([video, padding], dim=0)
        
        padded_videos.append(video)
        
        # Create agent mask: True for valid agents, False for padding
        mask = torch.zeros(max_agents, dtype=torch.bool)
        mask[:num_agents] = True
        agent_masks.append(mask)
    
    collated['video'] = torch.stack(padded_videos, dim=0)  # [B, max_A, ...]
    collated['agent_mask'] = torch.stack(agent_masks, dim=0)  # [B, max_A]
    collated['num_agents'] = torch.tensor([item['num_agents'] for item in batch])  # [B]
    
    # Handle other keys
    for key in batch[0].keys():
        if key in ['video', 'num_agents']:
            continue
        
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'agent_ids']:
            # Keep string/list values as lists
            collated[key] = values
        else:
            # For tensors, use default collate
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


def downstream_collate_fn(batch):
    """
    Collate function for downstream dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video tensor [T, C, H, W]
            - 'label': Task-specific label (shape depends on task)
            - metadata fields
    
    Returns:
        Dictionary with batched tensors
    """
    collated = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'match_id', 'player_id']:
            # Keep string values as lists
            collated[key] = values
        else:
            # Stack tensors
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated
