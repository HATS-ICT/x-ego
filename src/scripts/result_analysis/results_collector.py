#!/usr/bin/env python3
"""
Results Collector

Collects test results from all experiment directories and organizes them
by task, checkpoint type, and model type.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml


class ResultsCollector:
    """Collect and organize test results from experiment directories."""
    
    def __init__(self, output_dir: Path, task_definitions_path: Path):
        """
        Initialize results collector.
        
        Args:
            output_dir: Path to output directory containing experiment folders
            task_definitions_path: Path to task_definitions.csv
        """
        self.output_dir = Path(output_dir)
        self.task_definitions_path = Path(task_definitions_path)
        
        # Load task definitions
        self.task_definitions = pd.read_csv(task_definitions_path)
        self.task_info = {
            row['task_id']: {
                'task_name': row['task_name'],
                'category': row['category'],
                'ml_form': row['ml_form'],
                'temporal_type': row['temporal_type'],
                'horizon_sec': row['horizon_sec']
            }
            for _, row in self.task_definitions.iterrows()
        }
    
    def _read_hparam(self, exp_dir: Path) -> Optional[Dict]:
        """
        Read hparam.yaml from experiment directory.
        
        Args:
            exp_dir: Path to experiment directory
        
        Returns:
            Dictionary with hparam contents, or None if not found
        """
        hparam_path = exp_dir / 'hparam.yaml'
        if hparam_path.exists():
            with open(hparam_path, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    def _is_finetuned(self, hparam: Optional[Dict]) -> bool:
        """
        Determine if experiment is finetuned based on hparam.
        
        Args:
            hparam: Dictionary from hparam.yaml
        
        Returns:
            True if finetuned (has stage1_checkpoint), False if baseline
        """
        if hparam is None:
            return False
        model_cfg = hparam.get('model', {})
        stage1_checkpoint = model_cfg.get('stage1_checkpoint')
        return stage1_checkpoint is not None
    
    def _get_ui_mask(self, hparam: Optional[Dict]) -> str:
        """
        Get ui_mask setting from hparam.
        
        Args:
            hparam: Dictionary from hparam.yaml
        
        Returns:
            ui_mask value or 'unknown'
        """
        if hparam is None:
            return 'unknown'
        data_cfg = hparam.get('data', {})
        return data_cfg.get('ui_mask', 'unknown')
    
    def _is_new_format(self, exp_name: str) -> bool:
        """
        Check if experiment name matches the new format.
        
        New format: probe-{model}-{task_id}-{ui_mask}-{date}-{time}-{hash}
        Example: probe-dinov2-enemy_location_0s-all-260124-121450-pp1i
        
        Old format: probe-{task_id}-{model}-{date}-{time}-{hash}
        Example: probe-enemy_aliveCount-siglip2-260124-074908-h12m
        
        Args:
            exp_name: Experiment directory name
        
        Returns:
            True if new format, False otherwise
        """
        KNOWN_MODELS = {'clip', 'dinov2', 'siglip2', 'vjepa2'}
        
        parts = exp_name.split('-')
        if len(parts) < 5:
            return False
        
        # New format has model as the second part (parts[1])
        return parts[1] in KNOWN_MODELS

    def collect_all_results(self, exclude_folders: Optional[set] = None) -> Dict[str, Dict]:
        """
        Collect all test results from experiment directories.
        
        Args:
            exclude_folders: Set of folder names to exclude (e.g., {'pre-icml'})
        
        Returns:
            Dictionary mapping experiment names to their results
        """
        if exclude_folders is None:
            exclude_folders = {'pre-icml'}
        
        all_results = {}
        
        for exp_dir in self.output_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            if exp_dir.name in exclude_folders:
                continue
            
            # Only process folders starting with 'probe-'
            if not exp_dir.name.startswith('probe-'):
                continue
            
            # Only process new format: probe-{model}-{task}-{ui_mask}-{time}-{hash}
            if not self._is_new_format(exp_dir.name):
                continue
            
            # Parse experiment name to get task_id
            parsed = self.parse_experiment_name(exp_dir.name)
            task_id = parsed['task_id']
            
            # Skip if task_id is not in task_definitions.csv
            if task_id not in self.task_info:
                continue
            
            # Look for test results files
            best_results = exp_dir / 'test_results_best.json'
            last_results = exp_dir / 'test_results_last.json'
            
            exp_results = {}
            
            if best_results.exists():
                with open(best_results) as f:
                    exp_results['best'] = json.load(f)
            
            if last_results.exists():
                with open(last_results) as f:
                    exp_results['last'] = json.load(f)
            
            if exp_results:
                # Read hparam.yaml to determine baseline vs finetuned
                hparam = self._read_hparam(exp_dir)
                exp_results['is_finetuned'] = self._is_finetuned(hparam)
                exp_results['ui_mask'] = self._get_ui_mask(hparam)
                all_results[exp_dir.name] = exp_results
        
        return all_results
    
    def parse_experiment_name(self, exp_name: str) -> Dict[str, str]:
        """
        Parse experiment name to extract metadata.
        
        Two formats exist:
        1. probe-{model}-{task_id}-{ui_mask}-{date}-{time}-{hash}
           Example: probe-dinov2-enemy_location_0s-all-260124-121450-pp1i
        2. probe-{task_id}-{model}-{date}-{time}-{hash} (older format, no ui_mask)
           Example: probe-enemy_aliveCount-siglip2-260124-074908-h12m
        
        Models: clip, dinov2, siglip2, vjepa2
        UI masks: all, minimap_only, none
        
        Args:
            exp_name: Experiment directory name
        
        Returns:
            Dictionary with parsed components
        """
        KNOWN_MODELS = {'clip', 'dinov2', 'siglip2', 'vjepa2'}
        KNOWN_UI_MASKS = {'all', 'minimap_only', 'none'}
        
        parts = exp_name.split('-')
        
        if len(parts) < 5:
            return {
                'model_type': 'unknown',
                'task_id': 'unknown',
                'ui_mask': 'unknown',
                'timestamp': 'unknown',
                'hash': 'unknown'
            }
        
        # Last 3 parts are always: date, time, hash
        hash_val = parts[-1]
        timestamp = f"{parts[-3]}-{parts[-2]}"
        
        # Check if parts[1] is a known model (format 1) or task prefix (format 2)
        if parts[1] in KNOWN_MODELS:
            # Format 1: probe-{model}-{task_id}-{ui_mask}-{date}-{time}-{hash}
            model_type = parts[1]
            ui_mask = parts[-4] if parts[-4] in KNOWN_UI_MASKS else 'unknown'
            # Task ID is between model and ui_mask
            if ui_mask != 'unknown':
                task_id = '-'.join(parts[2:-4])
            else:
                task_id = '-'.join(parts[2:-3])
        else:
            # Format 2: probe-{task_id}-{model}-{date}-{time}-{hash}
            # Find the model in the parts
            model_type = 'unknown'
            model_idx = -1
            for i, part in enumerate(parts[1:-3]):
                if part in KNOWN_MODELS:
                    model_type = part
                    model_idx = i + 1  # +1 because we started from parts[1]
                    break
            
            if model_idx > 0:
                task_id = '-'.join(parts[1:model_idx])
            else:
                task_id = parts[1]
            ui_mask = 'unknown'  # This format doesn't have ui_mask in name
        
        # Convert dashes back to underscores in task_id
        task_id = task_id.replace('-', '_')
        
        return {
            'model_type': model_type,
            'task_id': task_id,
            'ui_mask': ui_mask,
            'timestamp': timestamp,
            'hash': hash_val
        }
    
    def organize_by_task_and_checkpoint(
        self, 
        all_results: Dict[str, Dict]
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Organize results by task ID and checkpoint type.
        
        Args:
            all_results: Raw results from collect_all_results()
        
        Returns:
            Nested dict: {task_id: {checkpoint_type: {exp_name: result}}}
        """
        organized = {}
        
        for exp_name, exp_results in all_results.items():
            parsed = self.parse_experiment_name(exp_name)
            task_id = parsed['task_id']
            
            if task_id not in organized:
                organized[task_id] = {'best': {}, 'last': {}}
            
            for checkpoint_type in ['best', 'last']:
                if checkpoint_type in exp_results:
                    organized[task_id][checkpoint_type][exp_name] = exp_results[checkpoint_type]
        
        return organized
    
    def create_results_dataframe(
        self, 
        all_results: Dict[str, Dict],
        checkpoint_type: str = 'best'
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from collected results.
        
        Args:
            all_results: Raw results from collect_all_results()
            checkpoint_type: 'best' or 'last'
        
        Returns:
            DataFrame with columns: exp_name, task_id, model_type, ui_mask, 
                                   is_finetuned, ml_form, category, temporal_type, 
                                   horizon_sec, metrics...
        """
        rows = []
        
        for exp_name, exp_results in all_results.items():
            if checkpoint_type not in exp_results:
                continue
            
            result = exp_results[checkpoint_type]
            parsed = self.parse_experiment_name(exp_name)
            task_id = result['task_id']
            
            # Get task info
            task_info = self.task_info.get(task_id, {})
            
            # Get is_finetuned and ui_mask from collected results (read from hparam.yaml)
            is_finetuned = exp_results.get('is_finetuned', False)
            ui_mask = exp_results.get('ui_mask', parsed.get('ui_mask', 'unknown'))
            
            row = {
                'exp_name': exp_name,
                'task_id': task_id,
                'model_type': parsed['model_type'],
                'ui_mask': ui_mask,
                'is_finetuned': is_finetuned,
                'init_type': 'finetuned' if is_finetuned else 'baseline',
                'ml_form': result['ml_form'],
                'num_classes': result.get('num_classes'),
                'output_dim': result.get('output_dim'),
                'category': task_info.get('category', 'unknown'),
                'temporal_type': task_info.get('temporal_type', 'unknown'),
                'horizon_sec': task_info.get('horizon_sec', -1),
            }
            
            # Add all metrics
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            # For multi_label_cls: convert hamming_dist to hamming_acc (1 - hamming_dist)
            if result['ml_form'] == 'multi_label_cls' and 'hamming_dist' in row:
                row['hamming_acc'] = 1.0 - row['hamming_dist']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by ml_form, then task_id, then model_type, then init_type
        if not df.empty:
            df = df.sort_values(['ml_form', 'task_id', 'model_type', 'ui_mask', 'init_type']).reset_index(drop=True)
        
        return df
    
    def get_results_by_ml_form(
        self, 
        all_results: Dict[str, Dict],
        checkpoint_type: str = 'best'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get results grouped by ML form.
        
        Args:
            all_results: Raw results from collect_all_results()
            checkpoint_type: 'best' or 'last'
        
        Returns:
            Dictionary mapping ml_form to DataFrame
        """
        df = self.create_results_dataframe(all_results, checkpoint_type)
        
        if df.empty:
            return {}
        
        grouped = {}
        for ml_form in df['ml_form'].unique():
            grouped[ml_form] = df[df['ml_form'] == ml_form].copy()
        
        return grouped


if __name__ == '__main__':
    # Example usage
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'output'
    task_defs = project_root / 'data' / 'labels' / 'task_definitions.csv'
    
    collector = ResultsCollector(output_dir, task_defs)
    all_results = collector.collect_all_results()
    
    print(f"Collected results from {len(all_results)} experiments")
    
    # Create dataframes
    df_best = collector.create_results_dataframe(all_results, 'best')
    df_last = collector.create_results_dataframe(all_results, 'last')
    
    print(f"\nBest checkpoint: {len(df_best)} results")
    print(f"Last checkpoint: {len(df_last)} results")
    
    # Group by ML form
    by_ml_form_best = collector.get_results_by_ml_form(all_results, 'best')
    print(f"\nML forms found: {list(by_ml_form_best.keys())}")
