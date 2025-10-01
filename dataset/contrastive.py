import logging
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

try:
    from .base_video_dataset import BaseVideoDataset
except ImportError:
    from base_video_dataset import BaseVideoDataset

try:
    from utils.dataset_utils import get_random_segment, get_team_voice_audio_clip_from_video_path, get_team_transcription_path_from_video_path, extract_text_segments_from_transcription, apply_minimap_mask
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_utils import get_team_voice_audio_clip_from_video_path, get_team_transcription_path_from_video_path, extract_text_segments_from_transcription

# TODO: generate_segment_description_seconds function is not found - may need to be implemented or removed

from decord import AudioReader, cpu

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def video_audio_text_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in ['video_path']:
            collated[key] = values
        elif key == 'raw_text':
            # raw_text is now a list of text clips per sample
            # Keep as list of lists for batch processing
            collated[key] = values
        else:
            collated[key] = torch.utils.data.default_collate(values)
    return collated

class CTFMContrastiveDataset(BaseVideoDataset, Dataset):
    def __init__(
        self,
        config: Dict,
        collect_timing: bool = False,
    ):
        # Store config for video processor initialization
        self._config = config
        
        # Initialize base class first
        super().__init__(config)
        
        data_config = config['data']
        self.path_config = config["path"]
        self.modalities = config.model.modalities
        
        self.return_video = 'video' in self.modalities
        self.return_audio = 'audio' in self.modalities
        self.return_text = 'text' in self.modalities
        
        self.audio_processor = None
        self.text_tokenizer = None
        
        self.data_path_csv_path = str(Path(self.path_config['data']) / data_config['data_path_csv_filename'])
        
        self.video_column_name = data_config['video_column_name']
        self.audio_sample_rate = data_config['audio_sample_rate']
        self.audio_num_channels = data_config['audio_num_channels']
        self.sample_audio_with_team_voice = data_config['sample_audio_with_team_voice']
        self.team_voice_audio_clip_weight = data_config['team_voice_audio_clip_weight']
        self.recording_text_trajectory_weight = data_config['recording_text_trajectory_weight']
        self.recording_text_transcription_weight = data_config['recording_text_transcription_weight']
        self.max_text_per_clip = data_config.max_text_per_clip
        self.communication_types_to_exclude = data_config.communication_types_to_exclude
        
        self.df = pd.read_csv(self.data_path_csv_path, keep_default_na=False)
        
        self.partition = data_config['partition']
        self.youtube_partition = data_config['youtube_partition']
        
        if self.partition != 'all':
            type_name = data_config['youtube_folder'] if self.partition == 'youtube' else data_config['recording_folder']
            initial_count = len(self.df)
            self.df = self.df[self.df['type'] == type_name].reset_index(drop=True)
            filtered_count = len(self.df)
            logger.info(f"Filtered dataset from {initial_count} to {filtered_count} samples for partition '{self.partition}'")
        
        # Apply youtube_partition filter if specified
        if self.partition in ['all', 'youtube'] and self.youtube_partition != 'all':
            initial_count = len(self.df)
            # Filter YouTube videos based on keywords in path
            if self.youtube_partition == 'tournament':
                # Include videos with 'tournament' in the path
                mask = (self.df['type'] == data_config['youtube_folder']) & (
                    self.df['video_path'].str.contains('tournament', case=False)
                )
            elif self.youtube_partition == 'tutorial':
                # Include videos with 'tutorial' in the path
                mask = (self.df['type'] == data_config['youtube_folder']) & (
                    self.df['video_path'].str.contains('tutorial', case=False)
                )
            else:
                raise ValueError(f"Invalid youtube_partition: {self.youtube_partition}. Must be 'all', 'tournament', or 'tutorial'")
            
            # Apply the filter only to youtube videos, keep all other videos
            youtube_filtered = self.df[mask]
            non_youtube = self.df[self.df['type'] != data_config['youtube_folder']]
            self.df = pd.concat([youtube_filtered, non_youtube], ignore_index=True)
            filtered_count = len(self.df)
            logger.info(f"Applied youtube_partition filter '{self.youtube_partition}': {initial_count} -> {filtered_count} samples")
        
        self.video_paths = self.df[self.video_column_name].tolist()
        self.video_types = self.df['type'].tolist()
        
        # Read split information from CSV (split column must exist)
        self.video_splits = self.df['split'].tolist()
        
        self.video_folder = data_config['video_folder']
        self.trajectory_folder = data_config['trajectory_folder']
        self.transcription_folder = data_config['transcription_folder']
        
        self.data_root = Path(self.path_config['data'])
        
        # Override video processor initialization if video is not needed
        if not self.return_video:
            self.video_processor = None
        
        if self.return_audio:
            self.audio_processor = AutoProcessor.from_pretrained(config.model.encoder.audio.processor_model).feature_extractor
            
        if self.return_text:
            self.text_tokenizer = AutoTokenizer.from_pretrained(config.model.encoder.text.processor_model)
            
        self.collect_timing = collect_timing
        
        self._epoch = 0
        self._sample_index = 0
        self._samples_seen = 0
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def filter_by_split(self, split: str):
        """Filter dataset to only include samples from a specific split."""
        split_indices = [i for i, s in enumerate(self.video_splits) if s == split]
        
        if not split_indices:
            raise ValueError(f"No samples found for split '{split}'. Available splits: {set(self.video_splits)}")
        self.video_paths = [self.video_paths[i] for i in split_indices]
        self.video_types = [self.video_types[i] for i in split_indices]
        self.video_splits = [self.video_splits[i] for i in split_indices]
        
        self.df = self.df.iloc[split_indices].reset_index(drop=True)
        return self
    
    def _sample_text_clip(self, video_path: str, video_type: str, start_seconds: float, end_seconds: float) -> List[str]:
        """Sample multiple text clips for multi-positive sampling."""
        all_text_clips = []
        
        if video_type == "recording":
            # For recordings, we can get both trajectory and transcription texts
            trajectory_clips = self._get_text_clip_from_recording_trajectory(video_path, start_seconds, end_seconds)
            transcription_clips = self._get_text_clip_from_recording_transcription(video_path, start_seconds, end_seconds)
            all_text_clips.extend(trajectory_clips)
            all_text_clips.extend(transcription_clips)
        elif video_type in ["youtube", "youtube_chunk"]:
            youtube_clips = self._get_text_clip_from_youtube(video_path, start_seconds, end_seconds)
            all_text_clips.extend(youtube_clips)
        else:
            raise ValueError(f"Invalid video type: {video_type}")
        
        # Remove empty strings and duplicates while preserving order
        unique_clips = []
        seen = set()
        for clip in all_text_clips:
            if clip and clip.strip() and clip not in seen:
                unique_clips.append(clip)
                seen.add(clip)
        
        # If we have fewer clips than max_text_per_clip, duplicate them
        if len(unique_clips) == 0:
            return [""] * self.max_text_per_clip
        elif len(unique_clips) < self.max_text_per_clip:
            # Duplicate clips to reach max_text_per_clip
            result = []
            while len(result) < self.max_text_per_clip:
                result.extend(unique_clips)
            return result[:self.max_text_per_clip]
        else:
            # Take first max_text_per_clip clips
            return unique_clips[:self.max_text_per_clip]
    
    def _get_text_clip_from_recording_trajectory(self, video_path: str, start_seconds: float, end_seconds: float) -> List[str]:
        """Get trajectory text clips from recording."""
        if self.recording_text_trajectory_weight <= 0:
            return []
        
        trajectory_path = video_path.replace(".mp4", ".csv").replace(self.video_folder, self.trajectory_folder)
        try:
            return generate_segment_description_seconds(trajectory_path, start_seconds, end_seconds, num_variants=20)
        except Exception:
            return []
    
    def _get_text_clip_from_recording_transcription(self, video_path: str, start_seconds: float, end_seconds: float, allow_overflow: bool = True) -> List[str]:
        """Get transcription text clips from recording."""
        if self.recording_text_transcription_weight <= 0:
            return []
            
        transcription_path = get_team_transcription_path_from_video_path(video_path, self.transcription_folder)
        if not Path(transcription_path).exists():
            return []
        
        try:
            transcription_text = extract_text_segments_from_transcription(transcription_path, start_seconds, end_seconds, exclude_labels=self.communication_types_to_exclude, allow_overflow=allow_overflow)
            return [transcription_text] if transcription_text else []
        except Exception as e:
            logger.warning(f"Failed to get transcription text from {transcription_path}: {e}")
            return []

    def _get_text_clip_from_youtube(self, video_path: str, start_seconds: float, end_seconds: float, allow_overflow: bool = True) -> List[str]:
        transcription_path = video_path.replace(".mp4", ".json").replace(self.video_folder, self.transcription_folder)
        if not Path(transcription_path).exists():
            return [""]
        transcription_text = extract_text_segments_from_transcription(transcription_path, start_seconds, end_seconds, allow_overflow)
        return [transcription_text]
    
    
    
    
    def _sample_audio_clip(self, video_path: str, start_seconds: float, end_seconds: float) -> torch.Tensor:
        audio_decoder = AudioReader(video_path, ctx=cpu(0), sample_rate=self.audio_sample_rate, mono=(self.audio_num_channels == 1))
        start_sample = int(start_seconds * self.audio_sample_rate)
        end_sample = int(end_seconds * self.audio_sample_rate)
        total_samples = audio_decoder.shape[1]
        start_sample = max(0, min(start_sample, total_samples - 1))
        end_sample = max(start_sample + 1, min(end_sample, total_samples))
        audio_clip = audio_decoder[start_sample:end_sample].asnumpy()
        
        is_recording_video = 'recording' in video_path
        if self.sample_audio_with_team_voice and is_recording_video:
            team_voice_audio_clip_path = get_team_voice_audio_clip_from_video_path(video_path)

            if Path(team_voice_audio_clip_path).exists():
                try:
                    voice_decoder = AudioReader(team_voice_audio_clip_path, ctx=cpu(0), sample_rate=self.audio_sample_rate, mono=(self.audio_num_channels == 1))
                    voice_total_samples = voice_decoder.shape[1]
                    voice_start_sample = max(0, min(start_sample, voice_total_samples - 1))
                    voice_end_sample = max(voice_start_sample + 1, min(end_sample, voice_total_samples))
                    voice_clip = voice_decoder[voice_start_sample:voice_end_sample].asnumpy()
                    
                    # Handle length mismatch between audio_clip and voice_clip by padding with zeros
                    audio_length = audio_clip.shape[-1]
                    voice_length = voice_clip.shape[-1]
                    
                    if voice_length < audio_length:
                        # Pad voice_clip with zeros to match audio_clip length
                        pad_length = audio_length - voice_length
                        voice_clip = np.pad(voice_clip, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
                    elif audio_length < voice_length:
                        # Pad audio_clip with zeros to match voice_clip length
                        pad_length = voice_length - audio_length
                        audio_clip = np.pad(audio_clip, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
                    
                    mixed_audio = audio_clip + voice_clip * self.team_voice_audio_clip_weight
                    return mixed_audio
                except Exception as e:
                    logger.warning(f"Failed to load team voice audio from {team_voice_audio_clip_path}: {e}")
            else:
                # team voice audio clip not found, use game audio only
                pass
        
        return audio_clip
    
    
    def _tokenize_text(self, text_clips: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize multiple text clips with padding and attention mask.
        
        Args:
            text_clips: List of text strings to tokenize
            
        Returns:
            Tuple containing:
                - text_features: Tensor of shape (max_text_per_clip, max_length) with tokenized text
                - text_attention_mask: Tensor of shape (max_text_per_clip, max_length) with attention masks
        """
        # Handle problematic text input
        if not text_clips or (isinstance(text_clips, (list, tuple)) and len(text_clips) == 0):
            text_clips = [""] * self.max_text_per_clip
        elif not isinstance(text_clips, (list, tuple)):
            text_clips = [str(text_clips)]
            
        # Ensure we have exactly max_text_per_clip items
        if len(text_clips) > self.max_text_per_clip:
            text_clips = text_clips[:self.max_text_per_clip]
        elif len(text_clips) < self.max_text_per_clip:
            # Pad with empty strings
            text_clips = text_clips + [""] * (self.max_text_per_clip - len(text_clips))
        
        # Process each text clip
        all_text_features = []
        all_attention_masks = []
        
        for text_clip in text_clips:
            # Clean up the text
            if not text_clip or not str(text_clip).strip():
                text_clip = ""
            else:
                text_clip = str(text_clip).strip()
            
            text_processed = self.text_tokenizer(
                text_clip, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            all_text_features.append(text_processed.input_ids.squeeze(0))
            all_attention_masks.append(text_processed.attention_mask.squeeze(0))
        
        # Stack all features and masks
        text_features = torch.stack(all_text_features, dim=0)  # Shape: (max_text_per_clip, max_length)
        text_attention_mask = torch.stack(all_attention_masks, dim=0)  # Shape: (max_text_per_clip, max_length)
        
        return text_features, text_attention_mask

    
    
    def _transform_audio(self, audio_clip: torch.Tensor) -> torch.Tensor:
        audio_processed = self.audio_processor(audio_clip, sampling_rate=self.audio_sample_rate, return_attention_mask=True, return_tensors="pt")
        audio_features = audio_processed.input_features.squeeze(0)
        audio_attention_mask = audio_processed.attention_mask.squeeze(0)
        return audio_features, audio_attention_mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample containing video, audio, and text data.
        
        Returns:
            dict: Dictionary containing:
                - 'video': Video tensor with processed video features
                - 'audio': Audio tensor with processed audio features
                - 'audio_attention_mask': Attention mask for audio
                - 'text': Text tensor with tokenized text features
                - 'text_attention_mask': Attention mask for text
                - 'video_path': Path to the video file
        """
        self._sample_index = idx
        self._samples_seen += 1
        
        video_path = self.video_paths[idx]
        video_type = self.video_types[idx]
        
        video_full_path = str(Path(self.path_config.data) / Path(video_path).relative_to("data"))
        
        # put placeholder values to be replaced later
        video_clip = torch.zeros(int(self.fixed_duration_seconds * self.target_fps), 3, 306, 544, dtype=torch.float16)
        audio_clip = torch.zeros(int(self.fixed_duration_seconds * self.audio_sample_rate))
        text_clips = [""] * self.max_text_per_clip
        
        result = {
            'video_path': video_path,
        }
        
        start_seconds, end_seconds, video_decoder = self._get_random_clip_range_from_video_metadata(video_full_path)
        
        # Handle video decoding failure
        if video_decoder is None:
            logger.warning(f"Skipping corrupted video {video_path}, using placeholder data")
            # Use placeholder values and skip to next sample
            # In a production system, you might want to retry with a different video
            start_seconds, end_seconds = 0.0, self.fixed_duration_seconds
        
        if self.return_video:
            try:
                if video_decoder is not None:
                    video_clip = self._sample_video_clip_from_decoder(video_decoder, start_seconds=start_seconds, end_seconds=end_seconds)
                else:
                    # Use placeholder video_clip
                    pass
            except Exception as e:
                print(f"Error processing video of {video_path}: {e}")
            video_features = self._transform_video(video_clip)
            result['video'] = video_features
            del video_clip
        
        if self.return_audio:
            try:
                if video_decoder is not None:
                    audio_clip = self._sample_audio_clip(video_full_path, start_seconds=start_seconds, end_seconds=end_seconds)
                else:
                    # Use placeholder audio_clip
                    pass
            except Exception as e:
                print(f"Error processing audio of {video_path}: {e}")
            audio_features, audio_attention_mask = self._transform_audio(audio_clip)
            result['audio'] = audio_features
            result['audio_attention_mask'] = audio_attention_mask
            del audio_clip
        
        if self.return_text:
            try:
                if video_decoder is not None:
                    text_clips = self._sample_text_clip(video_full_path, video_type, start_seconds=start_seconds, end_seconds=end_seconds)
                else:
                    # Use placeholder text_clips
                    text_clips = [""] * self.max_text_per_clip
            except Exception as e:
                print(f"Error processing text of {video_path}: {e}")
                text_clips = [""] * self.max_text_per_clip
            text_features, text_attention_mask = self._tokenize_text(text_clips)
            result['text'] = text_features
            result['text_attention_mask'] = text_attention_mask
            result['raw_text'] = text_clips
        return result
    
    def state_dict(self) -> Dict[str, int]:
        """
        Return the state dictionary for resumable training.
        
        Returns:
            dict: State dictionary containing epoch, sample index, and samples seen
        """
        return {
            'epoch': self._epoch,
            'sample_index': self._sample_index,
            'samples_seen': self._samples_seen
        }
    
    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """
        Load the state dictionary for resumable training.
        
        Args:
            state_dict: State dictionary containing epoch, sample index, and samples seen
        """
        self._epoch = state_dict['epoch']
        self._sample_index = state_dict['sample_index']
        self._samples_seen = state_dict['samples_seen']
        
        print(f"Loaded dataset state: epoch={self._epoch}, sample_index={self._sample_index}, samples_seen={self._samples_seen}")
    
    def on_epoch_start(self) -> None:
        """Called at the start of each epoch."""
        self._sample_index = 0
        
    def on_epoch_end(self) -> None:
        """Called at the end of each epoch."""
        self._epoch += 1
        self._sample_index = 0