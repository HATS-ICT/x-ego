import torch


def contrastive_collate_fn(batch):
    """
    Custom collate function for contrastive learning dataset.
    
    Concatenates all agents across the batch into the first dimension (no padding).
    Uses agent_counts to track how many agents belong to each sample for reconstruction.
    
    Args:
        batch: List of dictionaries containing:
            - 'videos': List of video tensors [T, C, H, W] per agent
            - 'videos_unmasked': List of unmasked video tensors (optional, for reconstruction)
            - 'num_agents': Number of alive agents in this sample
            - 'pov_team_side': String indicating team side
            - 'pov_team_side_encoded': Team side encoded as int
            - 'agent_ids': List of agent IDs
    
    Returns:
        Dictionary with:
            - 'video': Concatenated video tensor [total_agents, T, C, H, W]
            - 'video_unmasked': Concatenated unmasked video tensor (if present in batch)
            - 'agent_counts': Tensor [B] with number of agents per sample
            - Other metadata fields
    
    Example:
        If batch has 2 samples with [3, 2] agents respectively:
        - video shape: [5, T, C, H, W] (3 + 2 = 5 total agents)
        - agent_counts: tensor([3, 2])
    """
    collated = {}
    
    # Collect all agent videos and counts
    all_videos = []
    all_videos_unmasked = []
    agent_counts = []
    all_agent_ids = []
    
    # Check if unmasked videos are present
    has_unmasked = 'videos_unmasked' in batch[0]
    
    for item in batch:
        videos = item['videos']  # List of [T, C, H, W] tensors
        agent_counts.append(len(videos))
        all_videos.extend(videos)  # Flatten into single list
        all_agent_ids.extend(item['agent_ids'])
        
        if has_unmasked:
            all_videos_unmasked.extend(item['videos_unmasked'])
    
    # Stack all videos: [total_agents, T, C, H, W]
    collated['video'] = torch.stack(all_videos, dim=0)
    collated['agent_counts'] = torch.tensor(agent_counts, dtype=torch.long)  # [B]
    collated['agent_ids'] = all_agent_ids  # Flat list of all agent IDs
    
    # Stack unmasked videos if present (for reconstruction target)
    if has_unmasked:
        collated['video_unmasked'] = torch.stack(all_videos_unmasked, dim=0)
    
    # Handle other keys (per-sample metadata)
    collated['pov_team_side'] = [item['pov_team_side'] for item in batch]
    collated['pov_team_side_encoded'] = torch.stack(
        [item['pov_team_side_encoded'] for item in batch], dim=0
    )  # [B]
    collated['original_csv_idx'] = [item['original_csv_idx'] for item in batch]
    
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
