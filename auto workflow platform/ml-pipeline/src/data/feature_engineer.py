"""
Feature Engineer for FlowMind ML Pipeline

Creates advanced features for workflow prediction and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates advanced features for workflow ML models.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        self.fitted_models = {}
        
    def create_workflow_features(self, executions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create workflow-level aggregated features.
        
        Args:
            executions_df: Workflow executions DataFrame
            
        Returns:
            DataFrame with workflow features
        """
        try:
            # Group by workflow_id
            workflow_features = []
            
            for workflow_id, group in executions_df.groupby('workflow_id'):
                features = {
                    'workflow_id': workflow_id,
                    'workflow_name': group['workflow_name'].iloc[0] if 'workflow_name' in group.columns else f'workflow_{workflow_id}',
                    
                    # Execution count features
                    'total_executions': len(group),
                    'successful_executions': len(group[group['status'] == 'completed']),
                    'failed_executions': len(group[group['status'] == 'failed']),
                    'success_rate': len(group[group['status'] == 'completed']) / len(group),
                    
                    # Duration features
                    'avg_duration': group['duration'].mean(),
                    'median_duration': group['duration'].median(),
                    'std_duration': group['duration'].std(),
                    'min_duration': group['duration'].min(),
                    'max_duration': group['duration'].max(),
                    'duration_cv': group['duration'].std() / group['duration'].mean() if group['duration'].mean() > 0 else 0,
                    
                    # Trend features
                    'duration_trend': self._calculate_trend(group, 'duration'),
                    'success_trend': self._calculate_success_trend(group),
                    
                    # Frequency features
                    'executions_per_day': self._calculate_execution_frequency(group),
                    'most_common_hour': group['start_hour'].mode().iloc[0] if 'start_hour' in group.columns and not group['start_hour'].mode().empty else 0,
                    'weekend_execution_rate': self._calculate_weekend_rate(group),
                    
                    # Variability features
                    'execution_pattern_score': self._calculate_pattern_score(group),
                    'complexity_score': self._calculate_complexity_score(group)
                }
                
                workflow_features.append(features)
            
            features_df = pd.DataFrame(workflow_features)
            logger.info(f"Created features for {len(features_df)} workflows")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating workflow features: {e}")
            raise
            
    def create_execution_features(self, executions_df: pd.DataFrame, steps_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create execution-level features.
        
        Args:
            executions_df: Workflow executions DataFrame
            steps_df: Execution steps DataFrame (optional)
            
        Returns:
            DataFrame with execution features
        """
        try:
            exec_features = executions_df.copy()
            
            # Time-based features
            if 'start_time' in exec_features.columns:
                exec_features['execution_hour'] = exec_features['start_time'].dt.hour
                exec_features['execution_day_of_week'] = exec_features['start_time'].dt.dayofweek
                exec_features['is_weekend'] = (exec_features['start_time'].dt.dayofweek >= 5).astype(int)
                exec_features['is_business_hours'] = ((exec_features['start_time'].dt.hour >= 9) & 
                                                    (exec_features['start_time'].dt.hour < 17)).astype(int)
            
            # Duration-based features
            if 'duration' in exec_features.columns:
                # Relative to workflow average
                workflow_avg_duration = executions_df.groupby('workflow_id')['duration'].mean()
                exec_features['duration_vs_avg'] = exec_features.apply(
                    lambda row: row['duration'] / workflow_avg_duration.get(row['workflow_id'], 1) 
                    if workflow_avg_duration.get(row['workflow_id'], 0) > 0 else 1, axis=1
                )
                
                # Duration categories
                exec_features['duration_category'] = pd.cut(
                    exec_features['duration'], 
                    bins=[0, 60, 300, 1800, float('inf')], 
                    labels=['very_fast', 'fast', 'normal', 'slow']
                )
            
            # Add step-based features if steps data is available
            if steps_df is not None and len(steps_df) > 0:
                step_features = self._create_step_aggregations(steps_df)
                exec_features = exec_features.merge(
                    step_features, 
                    on='execution_id', 
                    how='left'
                )
            
            # JSON field features
            exec_features = self._extract_json_features(exec_features)
            
            logger.info(f"Created features for {len(exec_features)} executions")
            return exec_features
            
        except Exception as e:
            logger.error(f"Error creating execution features: {e}")
            raise
            
    def create_bottleneck_features(self, steps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for bottleneck detection.
        
        Args:
            steps_df: Execution steps DataFrame
            
        Returns:
            DataFrame with bottleneck-related features
        """
        try:
            bottleneck_features = []
            
            # Group by execution
            for execution_id, group in steps_df.groupby('execution_id'):
                group_sorted = group.sort_values('start_time')
                
                for idx, step in group_sorted.iterrows():
                    features = {
                        'step_id': step['step_id'],
                        'execution_id': execution_id,
                        'step_name': step['step_name'],
                        'step_type': step['step_type'],
                        'duration': step['duration'],
                        
                        # Position features
                        'step_position': list(group_sorted.index).index(idx),
                        'total_steps': len(group_sorted),
                        'is_first_step': list(group_sorted.index).index(idx) == 0,
                        'is_last_step': list(group_sorted.index).index(idx) == len(group_sorted) - 1,
                        
                        # Duration analysis
                        'duration_percentile_in_execution': self._calculate_percentile_rank(
                            step['duration'], group_sorted['duration']
                        ),
                        'duration_vs_execution_avg': step['duration'] / group_sorted['duration'].mean() if group_sorted['duration'].mean() > 0 else 1,
                        'duration_vs_execution_median': step['duration'] / group_sorted['duration'].median() if group_sorted['duration'].median() > 0 else 1,
                        
                        # Wait time features
                        'wait_time_before': self._calculate_wait_time(group_sorted, idx, 'before'),
                        'wait_time_after': self._calculate_wait_time(group_sorted, idx, 'after'),
                        
                        # Step type analysis
                        'step_type_avg_duration': self._get_step_type_avg_duration(steps_df, step['step_type']),
                        'step_type_frequency': self._get_step_type_frequency(steps_df, step['step_type']),
                        
                        # Resource utilization indicators
                        'performance_score': step.get('performance_score', 0),
                        'anomaly_score': step.get('anomaly_score', 0),
                        
                        # Error indicators
                        'has_error': 1 if pd.notna(step.get('error_message')) else 0,
                        'status_failed': 1 if step['status'] == 'failed' else 0
                    }
                    
                    bottleneck_features.append(features)
            
            features_df = pd.DataFrame(bottleneck_features)
            logger.info(f"Created bottleneck features for {len(features_df)} steps")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating bottleneck features: {e}")
            raise
            
    def create_prediction_features(self, executions_df: pd.DataFrame, lookback_window: int = 10) -> pd.DataFrame:
        """
        Create features for duration and performance prediction.
        
        Args:
            executions_df: Workflow executions DataFrame
            lookback_window: Number of previous executions to look back
            
        Returns:
            DataFrame with prediction features
        """
        try:
            prediction_features = []
            
            # Sort by workflow and time
            sorted_df = executions_df.sort_values(['workflow_id', 'start_time'])
            
            for workflow_id, group in sorted_df.groupby('workflow_id'):
                group_reset = group.reset_index(drop=True)
                
                for i in range(len(group_reset)):
                    current_exec = group_reset.iloc[i]
                    
                    # Historical features (from previous executions)
                    start_idx = max(0, i - lookback_window)
                    historical_data = group_reset.iloc[start_idx:i]
                    
                    features = {
                        'execution_id': current_exec['execution_id'],
                        'workflow_id': workflow_id,
                        
                        # Historical statistics
                        'historical_count': len(historical_data),
                        'avg_historical_duration': historical_data['duration'].mean() if len(historical_data) > 0 else 0,
                        'std_historical_duration': historical_data['duration'].std() if len(historical_data) > 0 else 0,
                        'last_duration': historical_data['duration'].iloc[-1] if len(historical_data) > 0 else 0,
                        
                        # Trend features
                        'duration_trend_slope': self._calculate_trend(historical_data, 'duration') if len(historical_data) >= 3 else 0,
                        'success_rate_recent': (historical_data['status'] == 'completed').mean() if len(historical_data) > 0 else 1,
                        
                        # Timing features
                        'time_since_last_execution': self._calculate_time_since_last(group_reset, i),
                        'execution_frequency_score': self._calculate_frequency_score(group_reset, i),
                        
                        # Pattern features
                        'same_hour_avg_duration': self._calculate_same_hour_avg(historical_data, current_exec),
                        'same_day_avg_duration': self._calculate_same_day_avg(historical_data, current_exec),
                        
                        # Current execution context
                        'current_hour': current_exec['start_time'].hour if 'start_time' in current_exec else 0,
                        'current_day_of_week': current_exec['start_time'].dayofweek if 'start_time' in current_exec else 0,
                        'is_weekend': 1 if current_exec['start_time'].dayofweek >= 5 else 0 if 'start_time' in current_exec else 0,
                    }
                    
                    prediction_features.append(features)
            
            features_df = pd.DataFrame(prediction_features)
            logger.info(f"Created prediction features for {len(features_df)} executions")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            raise
            
    def create_clustering_features(self, executions_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Create clustering-based features for workflow categorization.
        
        Args:
            executions_df: Workflow executions DataFrame
            n_clusters: Number of clusters for workflow grouping
            
        Returns:
            DataFrame with clustering features
        """
        try:
            # Prepare features for clustering
            cluster_features = []
            
            for workflow_id, group in executions_df.groupby('workflow_id'):
                features = [
                    group['duration'].mean(),
                    group['duration'].std(),
                    len(group),
                    (group['status'] == 'completed').mean(),
                    group['start_time'].dt.hour.mean() if 'start_time' in group.columns else 12,
                    (group['start_time'].dt.dayofweek >= 5).mean() if 'start_time' in group.columns else 0
                ]
                cluster_features.append(features)
            
            # Perform clustering
            cluster_data = np.array(cluster_features)
            cluster_data = np.nan_to_num(cluster_data)  # Handle NaN values
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(cluster_data)
            
            # Store the fitted model
            self.fitted_models['workflow_clustering'] = kmeans
            
            # Create cluster mapping
            workflow_ids = list(executions_df['workflow_id'].unique())
            cluster_mapping = dict(zip(workflow_ids, cluster_labels))
            
            # Add cluster information to executions
            executions_with_clusters = executions_df.copy()
            executions_with_clusters['workflow_cluster'] = executions_with_clusters['workflow_id'].map(cluster_mapping)
            
            # Add cluster-based features
            for cluster_id in range(n_clusters):
                cluster_mask = executions_with_clusters['workflow_cluster'] == cluster_id
                cluster_data = executions_with_clusters[cluster_mask]
                
                if len(cluster_data) > 0:
                    executions_with_clusters[f'cluster_{cluster_id}_avg_duration'] = cluster_data['duration'].mean()
                    executions_with_clusters[f'cluster_{cluster_id}_success_rate'] = (cluster_data['status'] == 'completed').mean()
                else:
                    executions_with_clusters[f'cluster_{cluster_id}_avg_duration'] = 0
                    executions_with_clusters[f'cluster_{cluster_id}_success_rate'] = 0
            
            logger.info(f"Created clustering features with {n_clusters} clusters")
            return executions_with_clusters
            
        except Exception as e:
            logger.error(f"Error creating clustering features: {e}")
            raise
            
    def _calculate_trend(self, data: pd.DataFrame, column: str) -> float:
        """Calculate trend slope for a numeric column."""
        if len(data) < 2:
            return 0
        
        x = np.arange(len(data))
        y = data[column].fillna(0)
        
        if len(y) < 2 or y.std() == 0:
            return 0
            
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0
        
    def _calculate_success_trend(self, data: pd.DataFrame) -> float:
        """Calculate success rate trend."""
        if len(data) < 2:
            return 0
            
        data_sorted = data.sort_values('start_time')
        success_values = (data_sorted['status'] == 'completed').astype(int)
        
        return self._calculate_trend(pd.DataFrame({'success': success_values}), 'success')
        
    def _calculate_execution_frequency(self, data: pd.DataFrame) -> float:
        """Calculate executions per day."""
        if len(data) < 2 or 'start_time' not in data.columns:
            return 0
            
        time_span = (data['start_time'].max() - data['start_time'].min()).days
        return len(data) / max(time_span, 1)
        
    def _calculate_weekend_rate(self, data: pd.DataFrame) -> float:
        """Calculate weekend execution rate."""
        if 'start_time' not in data.columns:
            return 0
            
        weekend_executions = (data['start_time'].dt.dayofweek >= 5).sum()
        return weekend_executions / len(data)
        
    def _calculate_pattern_score(self, data: pd.DataFrame) -> float:
        """Calculate execution pattern regularity score."""
        if len(data) < 3 or 'start_time' not in data.columns:
            return 0
            
        # Calculate time intervals between executions
        data_sorted = data.sort_values('start_time')
        intervals = data_sorted['start_time'].diff().dt.total_seconds().dropna()
        
        if len(intervals) < 2:
            return 0
            
        # Lower coefficient of variation indicates more regular pattern
        cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else float('inf')
        return 1 / (1 + cv)  # Normalize to 0-1 range
        
    def _calculate_complexity_score(self, data: pd.DataFrame) -> float:
        """Calculate workflow complexity score based on duration variability."""
        if 'duration' not in data.columns or len(data) < 2:
            return 0
            
        duration_cv = data['duration'].std() / data['duration'].mean() if data['duration'].mean() > 0 else 0
        return min(duration_cv, 5)  # Cap at 5 for extreme cases
        
    def _create_step_aggregations(self, steps_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated step features per execution."""
        step_aggs = steps_df.groupby('execution_id').agg({
            'duration': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'step_type': 'nunique',
            'performance_score': 'mean',
            'anomaly_score': 'mean'
        }).reset_index()
        
        # Flatten column names
        step_aggs.columns = ['execution_id', 'step_count', 'total_step_duration', 
                           'avg_step_duration', 'std_step_duration', 'min_step_duration',
                           'max_step_duration', 'unique_step_types', 'avg_performance_score',
                           'avg_anomaly_score']
        
        return step_aggs
        
    def _extract_json_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from JSON columns."""
        result_df = df.copy()
        
        json_columns = ['input_data', 'output_data', 'execution_path', 'bottlenecks']
        
        for col in json_columns:
            if col in df.columns:
                # Count of items in JSON
                result_df[f'{col}_count'] = df[col].apply(
                    lambda x: len(x) if isinstance(x, (dict, list)) else 0
                )
                
                # Check if JSON is empty
                result_df[f'{col}_is_empty'] = df[col].apply(
                    lambda x: 1 if not x or (isinstance(x, (dict, list)) and len(x) == 0) else 0
                )
        
        return result_df
        
    def _calculate_percentile_rank(self, value: float, series: pd.Series) -> float:
        """Calculate percentile rank of value in series."""
        return (series < value).mean()
        
    def _calculate_wait_time(self, group_sorted: pd.DataFrame, current_idx: int, direction: str) -> float:
        """Calculate wait time before or after a step."""
        if direction == 'before' and current_idx > 0:
            current_start = group_sorted.iloc[current_idx]['start_time']
            prev_end = group_sorted.iloc[current_idx - 1]['end_time']
            if pd.notna(current_start) and pd.notna(prev_end):
                return (current_start - prev_end).total_seconds()
        elif direction == 'after' and current_idx < len(group_sorted) - 1:
            current_end = group_sorted.iloc[current_idx]['end_time']
            next_start = group_sorted.iloc[current_idx + 1]['start_time']
            if pd.notna(current_end) and pd.notna(next_start):
                return (next_start - current_end).total_seconds()
        return 0
        
    def _get_step_type_avg_duration(self, steps_df: pd.DataFrame, step_type: str) -> float:
        """Get average duration for a step type across all executions."""
        type_data = steps_df[steps_df['step_type'] == step_type]
        return type_data['duration'].mean() if len(type_data) > 0 else 0
        
    def _get_step_type_frequency(self, steps_df: pd.DataFrame, step_type: str) -> float:
        """Get frequency of a step type."""
        return (steps_df['step_type'] == step_type).mean()
        
    def _calculate_time_since_last(self, group: pd.DataFrame, current_idx: int) -> float:
        """Calculate time since last execution."""
        if current_idx == 0:
            return 0
            
        current_time = group.iloc[current_idx]['start_time']
        last_time = group.iloc[current_idx - 1]['start_time']
        
        if pd.notna(current_time) and pd.notna(last_time):
            return (current_time - last_time).total_seconds()
        return 0
        
    def _calculate_frequency_score(self, group: pd.DataFrame, current_idx: int) -> float:
        """Calculate execution frequency score."""
        if current_idx < 3:
            return 0
            
        recent_data = group.iloc[max(0, current_idx-5):current_idx]
        if len(recent_data) < 2:
            return 0
            
        time_diffs = recent_data['start_time'].diff().dt.total_seconds().dropna()
        if len(time_diffs) == 0:
            return 0
            
        avg_interval = time_diffs.mean()
        return 1 / (1 + avg_interval / 3600)  # Normalize by hours
        
    def _calculate_same_hour_avg(self, historical_data: pd.DataFrame, current_exec: pd.Series) -> float:
        """Calculate average duration for same hour executions."""
        if len(historical_data) == 0 or 'start_time' not in current_exec:
            return 0
            
        current_hour = current_exec['start_time'].hour
        same_hour_data = historical_data[historical_data['start_time'].dt.hour == current_hour]
        
        return same_hour_data['duration'].mean() if len(same_hour_data) > 0 else 0
        
    def _calculate_same_day_avg(self, historical_data: pd.DataFrame, current_exec: pd.Series) -> float:
        """Calculate average duration for same day of week executions."""
        if len(historical_data) == 0 or 'start_time' not in current_exec:
            return 0
            
        current_day = current_exec['start_time'].dayofweek
        same_day_data = historical_data[historical_data['start_time'].dt.dayofweek == current_day]
        
        return same_day_data['duration'].mean() if len(same_day_data) > 0 else 0
