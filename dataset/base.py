import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import numpy as np
from abc import ABC, abstractmethod
from transformers import AutoVideoProcessor
from decord import VideoReader, cpu

try:
    from utils.dataset_utils import get_random_segment, apply_minimap_mask
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_utils import get_random_segment, apply_minimap_mask

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseVideoDataset(ABC):
    """
    Base class for video datasets with common video loading and processing functionality.
    
    This class provides shared methods for:
    - Video processor initialization
    - Path conversion utilities
    - Video clip loading using decord with optional time jitter
    - Video transformation and processing
    - Random clip range generation
    
    Time Jitter:
        When time_jitter_max_seconds > 0, each sampled frame timestamp is randomly
        perturbed by ±time_jitter_max_seconds to add temporal augmentation during training.
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize base video dataset with common configuration.
        
        Args:
            cfg: Configuration dictionary containing dataset parameters
        """
        self.data_cfg = cfg.data
        self.path_cfg = cfg.path
        
        # Common video parameters
        self.target_fps = self.data_cfg.target_fps
        self.fixed_duration_seconds = self.data_cfg.fixed_duration_seconds
        self.mask_minimap = self.data_cfg.mask_minimap
        self.time_jitter_max_seconds = self.data_cfg.time_jitter_max_seconds
        
        # Path configuration
        self.data_root = Path(cfg.path.data)
        
        # Initialize video processor
        self._init_video_processor()
    
    def _init_video_processor(self):
        """Initialize video processor based on configuration."""
        logger.debug(f"Initializing video processor with video_size_mode: {self.data_cfg.video_size_mode}")
        
        # Handle different config structures for processor model
        processor_model = None
        if hasattr(self.data_cfg, 'video_processor_model'):
            processor_model = self.data_cfg.video_processor_model
            
        elif hasattr(self, '_config') and hasattr(self._config, 'model') and hasattr(self._config.model, 'encoder') and hasattr(self._config.model.encoder, 'video'):
            processor_model = self._config.model.encoder.video.processor_model
        
        if processor_model is None:
            logger.warning("No video processor model found in config, video processor will not be initialized")
            self.video_processor = None
            return
        
        if self.data_cfg.video_size_mode == "resize_center_crop":
            self.video_processor = AutoVideoProcessor.from_pretrained(processor_model)
        elif self.data_cfg.video_size_mode == "resize_distort":
            self.video_processor = AutoVideoProcessor.from_pretrained(
                processor_model,
                do_center_crop=False
            )
            self.video_processor.size = {"width": 224, "height": 224}
        else:
            raise ValueError(f"Unsupported video_size_mode: {self.data_cfg.video_size_mode}")
    
    def _to_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path using configured data directory."""
        if not Path(path).is_absolute():
            # Remove data/ prefix if present and use configured data root
            path_obj = Path(path)
            if path_obj.parts[0] == "data":
                relative_path = Path(*path_obj.parts[1:])
                path = str(self.data_root / relative_path)
            else:
                if self.data_root:
                    path = str(self.data_root / path)
                elif self.path_cfg and hasattr(self.path_cfg, 'data'):
                    # Fallback for different path configurations
                    path = str(Path(self.path_cfg.data) / Path(path).relative_to("data"))
        return path
    
    def _get_random_clip_range_from_video_metadata(self, video_path: str) -> Tuple[Optional[float], Optional[float], Optional[VideoReader]]:
        """Get random clip range from video metadata using decord."""
        try:
            decoder = VideoReader(video_path, ctx=cpu(0))
            fps = decoder.get_avg_fps()
            total_frames = len(decoder)
            full_duration = total_frames / fps
            start_seconds, end_seconds = get_random_segment(full_duration, self.fixed_duration_seconds)
            return start_seconds, end_seconds, decoder
        except Exception as e:
            logger.error(f"Failed to decode video {video_path}: {e}")
            # Return None to indicate failure - will be handled in calling code
            return None, None, None
    
    def _sample_video_clip_from_decoder(self, video_decoder: VideoReader, start_seconds: float, end_seconds: float) -> torch.Tensor:
        """
        Sample video clip from an existing VideoReader (decord).
        
        Args:
            video_decoder: VideoReader instance
            start_seconds: Start time of the clip
            end_seconds: End time of the clip
            
        Returns:
            Video tensor of shape (num_frames, channels, height, width)
        """
        video_fps = video_decoder.get_avg_fps()
        
        # Calculate expected number of frames based on fixed duration and target fps
        if self.fixed_duration_seconds is not None:
            target_frames = int(self.fixed_duration_seconds * self.target_fps)
            actual_duration = self.fixed_duration_seconds
        else:
            actual_duration = end_seconds - start_seconds
            target_frames = int(actual_duration * self.target_fps)
        
        if actual_duration >= (self.fixed_duration_seconds - 1e-6) if self.fixed_duration_seconds else True:
            # Case 1: Enough video, sample exactly target_frames at target_fps
            duration_to_use = self.fixed_duration_seconds if self.fixed_duration_seconds else actual_duration
            timestamps = np.linspace(start_seconds, start_seconds + duration_to_use, target_frames, endpoint=False)
        else:
            # Case 2: Video too short - sample at target_fps for available duration, then pad
            available_frames = int(actual_duration * self.target_fps)
            if available_frames > 0:
                timestamps = np.linspace(start_seconds, end_seconds, available_frames, endpoint=False)
            else:
                # Handle edge case where duration is extremely short
                timestamps = np.array([start_seconds])
            
            # Pad to target length by repeating the last timestamp
            if len(timestamps) < target_frames:
                last_timestamp = timestamps[-1]
                pad_count = target_frames - len(timestamps)
                pad_timestamps = np.full(pad_count, last_timestamp)
                timestamps = np.concatenate([timestamps, pad_timestamps])
        
        # Apply time jitter if configured
        if self.time_jitter_max_seconds > 0:
            # Add random jitter to each timestamp
            jitter = np.random.uniform(-self.time_jitter_max_seconds, self.time_jitter_max_seconds, size=len(timestamps))
            timestamps = timestamps + jitter
            # Clamp timestamps to valid range [0, total_duration]
            total_duration = len(video_decoder) / video_decoder.get_avg_fps()
            timestamps = np.clip(timestamps, 0, total_duration)

        # Convert timestamps to frame indices in the original video
        frame_indices = (timestamps * video_fps).astype(int)
        
        # Ensure frame indices are within bounds
        max_frame_index = len(video_decoder) - 1
        frame_indices = np.clip(frame_indices, 0, max_frame_index)
        
        video_clip = video_decoder.get_batch(frame_indices.tolist()) 
        video_clip = torch.from_numpy(video_clip.asnumpy()).permute(0, 3, 1, 2).half() # from HWC to CHW format, half precision to save memory
        
        if self.mask_minimap:
            video_clip = apply_minimap_mask(video_clip)
        
        return video_clip
    
    def _load_video_clip_with_decord(self, video_path: str, start_seconds: float, end_seconds: float) -> torch.Tensor:
        """
        Load video clip using decord.
        
        Args:
            video_path: Path to the video file
            start_seconds: Start time of the clip
            end_seconds: End time of the clip
            
        Returns:
            Video tensor of shape (num_frames, channels, height, width)
        """
        # Calculate expected number of frames
        if self.fixed_duration_seconds is not None:
            expected_frames = int(self.fixed_duration_seconds * self.target_fps)
        else:
            expected_frames = int((end_seconds - start_seconds) * self.target_fps)
        
        # Convert to absolute path
        video_full_path = self._to_absolute_path(video_path)
        
        try:
            decoder = VideoReader(video_full_path, ctx=cpu(0))
            return self._sample_video_clip_from_decoder(decoder, start_seconds, end_seconds)
        except Exception as e:
            logger.warning(f"Failed to load video {video_path}: {e}, using placeholder data")
            # Create placeholder video with expected dimensions
            return torch.zeros(expected_frames, 3, 306, 544, dtype=torch.float16)
    
    def _transform_video(self, video_clip: torch.Tensor) -> torch.Tensor:
        """
        Transform video clip using the video processor.
        
        Args:
            video_clip: Video tensor of shape (num_frames, channels, height, width)
            
        Returns:
            video_features: Processed video features
        """
        # Process video through the processor
        video_processed = self.video_processor(video_clip, return_tensors="pt")
        video_features = video_processed.pixel_values_videos.squeeze(0)
        return video_features
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        pass
