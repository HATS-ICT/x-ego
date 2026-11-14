"""
Contrastive Learning Dataset for Team Alignment (Stage 1)

This dataset provides multi-agent video data for contrastive learning,
where positive pairs are agents from the same team at the same time,
and negative pairs are agents from different teams/times.

Three pairing modes (data.pair_mode) implement the P1 adaptation arms:
  cecl    -- positives are same-team, same-time player views (the intended supervision)
  single  -- positives are independent augmentations of ONE observer clip, so the
             group size and processed-view budget match `cecl` without any
             cross-player correspondence
  shuffle -- positives are nuisance-matched pseudo-groups read from a prebuilt
             manifest (same clip multiset as `cecl`, true correspondence removed)

Key features:
- Loads from contrastive.csv (no labels, just metadata and teammate info)
- Variable number of agents per sample (no padding for dead agents)
- Batch collation flattens all agents into first dimension with agent_counts tracking
- Example: batch with samples having [3, 2] agents -> video tensor [5, T, C, H, W]
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from rich import print as rprint

from .dataset_utils import (
    init_video_processor,
    construct_video_path,
    load_video_clip,
    transform_video,
)


# Pairing modes for the P1 adaptation arms (see module docstring).
PAIR_MODES = {"cecl", "single", "shuffle"}


def plot_video_example(
    video_clip: torch.Tensor,
    save_path: Union[str, Path],
) -> None:
    """
    Plot a video clip as a 4x5 grid of frames and save to file.

    Args:
        video_clip: Tensor of shape (20, 3, 306, 306) [T, C, H, W].
        save_path: Path to save the figure (e.g. .png). Will be saved in artifact/ folder.
    """
    save_path = Path(save_path)
    artifact_dir = Path("artifact")
    artifact_dir.mkdir(exist_ok=True)
    full_save_path = artifact_dir / save_path.name
    
    n_frames = video_clip.shape[0]
    if n_frames != 20:
        rprint(f"[yellow]WARN[/yellow] plot_video_example expects 20 frames, got [bold]{n_frames}[/bold]; grid may be incomplete")

    # [T, C, H, W] -> numpy, then [T, H, W, C] for imshow
    arr = video_clip.detach().float().numpy()
    if arr.max() > 1.0:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    arr = np.transpose(arr, (0, 2, 3, 1))  # [T, H, W, C]

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_frames:
            ax.imshow(arr[i])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
    plt.close()
    rprint(f"[green]OK[/green] Saved video example to [bold]{full_save_path}[/bold]")


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning (team alignment).
    
    Loads from contrastive.csv which contains:
    - Metadata: match_id, round_num, partition, timestamps, etc.
    - Teammate info: num_alive_teammates, teammate_0_id through teammate_4_id
      (alive agents are guaranteed to be the first k teammate_idx where k = num_alive_teammates)
    
    No labels are used - contrastive alignment is based on batch structure.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Contrastive Learning Dataset.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        self.cfg = cfg
        self.pair_mode = str(getattr(self.cfg.data, 'pair_mode', 'cecl'))
        if self.pair_mode not in PAIR_MODES:
            raise ValueError(
                f"Unsupported data.pair_mode: '{self.pair_mode}'. Expected one of {sorted(PAIR_MODES)}"
            )
        
        # Load label CSV file
        self.df = pl.read_csv(self.cfg.data.label_path, null_values=[])
        
        # Partition filtering
        if self.cfg.data.partition != 'all':
            initial_count = len(self.df)
            self.df = self.df.filter(pl.col('partition') == self.cfg.data.partition)
            filtered_count = len(self.df)
            rprint(f"[blue]->[/blue] Filtered dataset from [bold]{initial_count:,}[/bold] to [bold]{filtered_count:,}[/bold] samples for partition [cyan]'{self.cfg.data.partition}'[/cyan]")
        
        # Initialize video processor
        self._init_video_processor()
        
        # Load time offsets
        data_root = Path(self.cfg.path.data)
        map_name = self.cfg.data.map
        offset_path = data_root / map_name / "time_offset.json"
            
        if not offset_path.exists():
            raise FileNotFoundError(
                f"time_offset.json not found at {offset_path}. "
                f"Run compute_offset.py for map '{map_name}' first."
            )
        with open(offset_path, "r") as f:
            self.time_offsets = json.load(f)
        rprint(f"[green]OK[/green] Loaded time offsets from {offset_path.name}")
        
        # Shuffled arm: replace the team-row frame with prebuilt pseudo-groups.
        self.groups = None
        if self.pair_mode == 'shuffle':
            self._load_shuffle_manifest()
        
        rprint(
            f"[green]OK[/green] Contrastive dataset initialized with "
            f"[bold]{len(self.df):,}[/bold] groups (pair_mode=[cyan]{self.pair_mode}[/cyan])"
        )
    
    def _load_shuffle_manifest(self) -> None:
        """
        Load the prebuilt shuffled-pair assignment.

        The manifest is long-form (one row per clip). Groups are rebuilt here and
        `self.df` is replaced by a group-level frame so partition filtering and the
        video batch sampler keep working unchanged. `num_alive_teammates` carries
        the group size for that sampler.
        """
        manifest_name = getattr(self.cfg.data, 'shuffle_manifest', None)
        if not manifest_name:
            raise ValueError("data.pair_mode='shuffle' requires data.shuffle_manifest")
        
        manifest_path = Path(manifest_name)
        if not manifest_path.is_absolute():
            manifest_path = (
                Path(self.cfg.path.data) / self.cfg.data.map / self.cfg.data.labels_folder / manifest_name
            )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Shuffled-pair manifest not found: {manifest_path}. "
                f"Build it with src/scripts/iclr2027/build_shuffle_manifest.py"
            )
        
        members = pl.read_csv(manifest_path, null_values=[])
        required = {
            'group_id', 'partition', 'member_idx', 'src_idx', 'match_id',
            'round_num', 'agent_id', 'start_seconds', 'end_seconds',
            'pov_team_side', 'phase_bin',
        }
        missing = required - set(members.columns)
        if missing:
            raise ValueError(f"Shuffle manifest {manifest_path} is missing columns: {sorted(missing)}")
        
        members = members.sort(['group_id', 'member_idx'])
        self.groups = {}
        for group_id, block in members.group_by('group_id', maintain_order=True):
            key = int(group_id[0] if isinstance(group_id, tuple) else group_id)
            self.groups[key] = block.to_dicts()
        
        self.df = (
            members.group_by('group_id', maintain_order=True)
            .agg(
                pl.col('partition').first(),
                pl.col('pov_team_side').first(),
                pl.col('phase_bin').first(),
                pl.len().alias('num_alive_teammates'),
            )
            .rename({'group_id': 'idx'})
            .sort('idx')
        )
        rprint(
            f"[green]OK[/green] Loaded shuffled-pair manifest [dim]{manifest_path.name}[/dim]: "
            f"[bold]{len(self.df):,}[/bold] groups / [bold]{len(members):,}[/bold] clips"
        )
    
    def _init_video_processor(self):
        """Initialize video processor based on model type from config."""
        self.video_processor, self.processor_type = init_video_processor(self.cfg)
        self.model_type = self.cfg.model.encoder.model_type
    
    def _get_alive_agent_ids(self, row: Dict[str, Any]) -> List[str]:
        num_alive = int(row['num_alive_teammates'])
        agent_ids = []
        for i in range(num_alive):
            agent_id = row[f'teammate_{i}_id']
            agent_ids.append(str(agent_id))
        return agent_ids
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int | tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Get a sample for contrastive learning.
        
        Returns:
            dict: Dictionary containing:
                - 'videos': List of video tensors, one per agent [T, C, H, W] each
                - 'num_agents': Number of alive agents in this sample
                - 'pov_team_side': Team side (string)
                - 'pov_team_side_encoded': Team side encoded as int
                - 'agent_ids': List of agent IDs
                - 'original_csv_idx': Original CSV index for reference
        """
        max_agents = None
        if isinstance(idx, tuple):
            idx, max_agents = idx

        row = self.df.row(idx, named=True)
        clip_specs = self._resolve_clip_specs(row, max_agents)
        pov_team_side = str(row['pov_team_side']).upper()
        original_csv_idx = row['idx']
        
        agent_videos = []
        for spec in clip_specs:
            # Each spec is loaded independently, so the random tube mask is drawn
            # afresh per view. In 'single' mode that is exactly what makes the two
            # views of one clip independent augmentations.
            video_clip = self._load_clip(spec)
            
            ## Temporary: Debug, plot each video clip and save png
            # plot_video_example(video_clip, f"debug_video_{spec['agent_id']}.png")
            
            video_features = transform_video(self.video_processor, self.processor_type, video_clip)
            agent_videos.append(video_features)  # Each is [T, C, H, W]
            
        # Encode team side
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        # Return list of videos (not stacked) - collate will concatenate across batch
        result = {
            'videos': agent_videos,  # List of [T, C, H, W] tensors
            'num_agents': len(clip_specs),
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'agent_ids': [str(spec['agent_id']) for spec in clip_specs],
            'original_csv_idx': original_csv_idx,
        }
        
        return result
    
    def _resolve_clip_specs(self, row: Dict[str, Any], max_agents: int | None) -> List[Dict[str, Any]]:
        """
        Resolve one positive group into a list of clips to load.

        All three arms return the same group sizes for the same row, so the
        positive degree and the processed-view budget stay matched across arms.
        Only the identity of the clips differs.
        """
        if self.pair_mode == 'shuffle':
            members = self.groups[int(row['idx'])]
            if max_agents is not None:
                members = members[:max_agents]
            return [
                {
                    'match_id': member['match_id'],
                    'round_num': int(member['round_num']),
                    'agent_id': str(member['agent_id']),
                    'start_seconds': float(member['start_seconds']),
                    'end_seconds': float(member['end_seconds']),
                }
                for member in members
            ]
        
        agent_ids = self._get_alive_agent_ids(row)
        if max_agents is not None:
            agent_ids = agent_ids[:max_agents]
        
        if self.pair_mode == 'single':
            # Same group size, but every view comes from roster slot 0 of this row.
            # The slot rule is fixed and uses only information at the input window.
            agent_ids = [agent_ids[0]] * len(agent_ids)
        
        return [
            {
                'match_id': row['match_id'],
                'round_num': int(row['round_num']),
                'agent_id': str(agent_id),
                'start_seconds': float(row['start_seconds']),
                'end_seconds': float(row['end_seconds']),
            }
            for agent_id in agent_ids
        ]
    
    def _load_clip(self, spec: Dict[str, Any]) -> torch.Tensor:
        """Load one observer clip, applying its per-round synchronization offset."""
        match_id = spec['match_id']
        agent_id = spec['agent_id']
        round_num = spec['round_num']
        video_path = construct_video_path(self.cfg, match_id, agent_id, round_num)
        
        # Get offset and fail fast for missing entries.
        round_key = f"round_{round_num}"
        try:
            round_offsets = self.time_offsets[str(match_id)][str(agent_id)][round_key]
        except KeyError as exc:
            raise KeyError(
                f"No offset data for '{match_id}/{agent_id}/{round_key}'"
            ) from exc
        offset_sec = float(round_offsets["offset_sec"])
        
        video_result = load_video_clip(
            self.cfg,
            video_path,
            spec['start_seconds'] + offset_sec,
            spec['end_seconds'] + offset_sec,
        )
        return video_result['video']  # [T, C, H, W]
