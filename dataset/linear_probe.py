"""
Linear Probing Dataset (Stage 2)

Generic dataset for downstream tasks using precomputed embeddings.
Loads video embeddings and task-specific labels based on task_id.
"""

import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def linear_probe_collate_fn(batch):
    """
    Collate function for linear probing dataset.
    
    Args:
        batch: List of dictionaries containing:
            - 'video': Video embeddings [A, embed_dim]
            - 'label': Task-specific label (shape depends on task)
            - metadata fields
    
    Returns:
        Dictionary with batched tensors
    """
    collated = {}
    
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        
        if key in ['pov_team_side', 'match_id']:
            # Keep string values as lists
            collated[key] = values
        else:
            # Stack tensors
            collated[key] = torch.utils.data.default_collate(values)
    
    return collated


class LinearProbeDataset(Dataset):
    """
    Dataset for Stage 2 linear probing.
    
    Loads:
    - Precomputed video embeddings from h5 file
    - Task-specific labels from CSV
    
    Works with any task defined in task_definitions.csv.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize Linear Probe Dataset.
        
        Args:
            cfg: Configuration with:
                - task.task_id: Task identifier
                - task.ml_form: ML form (binary_cls, multi_cls, etc.)
                - task.label_column: Column name(s) for labels
                - data.labels_filename: Path to labels CSV
                - model.encoder.video.model_type: Encoder type for embeddings
        """
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.task_cfg = cfg.task
        
        # Task info
        self.task_id = self.task_cfg.task_id
        self.ml_form = self.task_cfg.ml_form
        self.output_dim = self.task_cfg.output_dim
        self.num_classes = self.task_cfg.num_classes
        
        # Load label CSV
        data_root = Path(cfg.path.data)
        label_path = data_root / self.data_cfg.labels_folder / self.data_cfg.labels_filename
        
        if not label_path.exists():
            raise FileNotFoundError(f"Labels file not found: {label_path}")
        
        self.df = pd.read_csv(label_path, keep_default_na=False)
        logger.info(f"Loaded {len(self.df)} samples from {label_path}")
        
        # Store original CSV index for embedding lookup
        self.df['original_csv_idx'] = self.df.index
        
        # Filter by partition
        partition = self.data_cfg.partition
        if partition != 'all':
            initial = len(self.df)
            self.df = self.df[self.df['partition'] == partition].reset_index(drop=True)
            logger.info(f"Filtered to {len(self.df)} samples for partition '{partition}'")
        
        # Load precomputed embeddings
        self.embeddings_encoder = self.data_cfg.precomputed_embeddings_encoder
        csv_stem = Path(self.data_cfg.labels_filename).stem
        embeddings_filename = f"{csv_stem}_{self.embeddings_encoder}.h5"
        embeddings_path = data_root / "video_544x306_30fps_embed" / embeddings_filename
        
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                f"Run embedding extraction first."
            )
        
        self.embeddings_h5 = h5py.File(embeddings_path, 'r')
        logger.info(f"Using precomputed embeddings: {embeddings_path}")
        
        # Number of agents
        self.num_agents = self.data_cfg.num_pov_agents
        
        # Parse label column configuration
        self._parse_label_columns()
        
        logger.info(f"Task: {self.task_id} ({self.ml_form})")
        logger.info(f"Output dim: {self.output_dim}, Num classes: {self.num_classes}")
    
    def _parse_label_columns(self):
        """Parse label column names from task config."""
        label_column = self.task_cfg.label_column
        
        # Label column can be a single string or semicolon-separated list
        if isinstance(label_column, str):
            if ';' in label_column:
                self.label_columns = label_column.split(';')
            else:
                self.label_columns = [label_column]
        else:
            self.label_columns = list(label_column)
        
        # Verify columns exist
        for col in self.label_columns:
            if col not in self.df.columns:
                raise ValueError(f"Label column '{col}' not found in CSV. Available: {list(self.df.columns)}")
        
        logger.info(f"Label columns: {self.label_columns}")
    
    def _load_embedding(self, csv_idx: int, agent_position: int) -> torch.Tensor:
        """Load precomputed embedding for an agent."""
        embedding = self.embeddings_h5[str(csv_idx)][str(agent_position)][:]
        return torch.from_numpy(embedding).float()
    
    def _load_all_agent_embeddings(self, csv_idx: int) -> torch.Tensor:
        """Load embeddings for all agents."""
        embeddings = []
        for agent_pos in range(self.num_agents):
            emb = self._load_embedding(csv_idx, agent_pos)
            embeddings.append(emb)
        return torch.stack(embeddings, dim=0)  # [A, embed_dim]
    
    def _get_label(self, row: pd.Series) -> torch.Tensor:
        """
        Extract label from row based on task type.
        
        Returns tensor of appropriate shape for the task.
        """
        if self.ml_form == 'binary_cls':
            # Single binary value
            value = row[self.label_columns[0]]
            return torch.tensor(float(value), dtype=torch.float32)
        
        elif self.ml_form == 'multi_cls':
            # Single class index
            value = row[self.label_columns[0]]
            return torch.tensor(int(value), dtype=torch.long)
        
        elif self.ml_form == 'multi_label_cls':
            # Multi-hot vector - label column should contain the multi-hot encoding
            # or we need to construct it from multiple columns
            if len(self.label_columns) == 1:
                # Single column with multi-hot or needs parsing
                col = self.label_columns[0]
                value = row[col]
                
                # Check if it's already a list/array stored as string
                if isinstance(value, str) and value.startswith('['):
                    import ast
                    value = ast.literal_eval(value)
                    return torch.tensor(value, dtype=torch.float32)
                else:
                    # Assume it's a single value to be one-hot encoded
                    label = torch.zeros(self.num_classes, dtype=torch.float32)
                    label[int(value)] = 1.0
                    return label
            else:
                # Multiple columns - each column is a binary indicator
                values = [float(row[col]) for col in self.label_columns]
                return torch.tensor(values, dtype=torch.float32)
        
        elif self.ml_form == 'regression':
            # Continuous value(s)
            if len(self.label_columns) == 1:
                value = row[self.label_columns[0]]
                return torch.tensor(float(value), dtype=torch.float32)
            else:
                values = [float(row[col]) for col in self.label_columns]
                return torch.tensor(values, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unknown ml_form: {self.ml_form}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        row = self.df.iloc[idx]
        original_csv_idx = row['original_csv_idx']
        
        # Load embeddings
        video = self._load_all_agent_embeddings(original_csv_idx)
        
        # Get label
        label = self._get_label(row)
        
        # Get team side
        pov_team_side = row.get('pov_team_side', 'unknown')
        if isinstance(pov_team_side, str):
            pov_team_side = pov_team_side.upper()
        pov_team_side_encoded = 1 if pov_team_side == 'CT' else 0
        
        return {
            'video': video,
            'label': label,
            'pov_team_side': pov_team_side,
            'pov_team_side_encoded': torch.tensor(pov_team_side_encoded, dtype=torch.long),
            'original_csv_idx': original_csv_idx,
        }
    
    def close(self):
        """Close h5 file handle."""
        if self.embeddings_h5 is not None:
            self.embeddings_h5.close()
    
    def __del__(self):
        self.close()


if __name__ == "__main__":
    print("LinearProbeDataset test placeholder")
