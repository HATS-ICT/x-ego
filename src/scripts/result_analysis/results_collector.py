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
                all_results[exp_dir.name] = exp_results
        
        return all_results
    
    def parse_experiment_name(self, exp_name: str) -> Dict[str, str]:
        """
        Parse experiment name to extract metadata.
        
        Expected format: probe-{model}-{task_id}-{contra_init}-{timestamp}-{hash}
        Example: probe-siglip2-self_location_0s-none-260122-053730-lqeq
        
        Args:
            exp_name: Experiment directory name
        
        Returns:
            Dictionary with parsed components
        """
        parts = exp_name.split('-')
        
        if len(parts) < 6:
            return {
                'model_type': 'unknown',
                'task_id': 'unknown',
                'contra_init': 'unknown',
                'timestamp': 'unknown',
                'hash': 'unknown'
            }
        
        # Handle task_id which may contain underscores
        # Format: probe-model-task_id-contra_init-timestamp-hash
        model_type = parts[1]
        contra_init = parts[-3]
        timestamp = f"{parts[-2]}-{parts[-1]}"
        hash_val = parts[-1]
        
        # Task ID is everything between model and contra_init
        task_id = '-'.join(parts[2:-3])
        
        return {
            'model_type': model_type,
            'task_id': task_id,
            'contra_init': contra_init,
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
            DataFrame with columns: exp_name, task_id, model_type, ml_form, 
                                   category, temporal_type, horizon_sec, metrics...
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
            
            row = {
                'exp_name': exp_name,
                'task_id': task_id,
                'model_type': parsed['model_type'],
                'contra_init': parsed['contra_init'],
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
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by ml_form, then task_id
        if not df.empty:
            df = df.sort_values(['ml_form', 'task_id']).reset_index(drop=True)
        
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
