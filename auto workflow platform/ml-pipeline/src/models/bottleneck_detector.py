"""
Bottleneck Detector for FlowMind ML Pipeline

Identifies bottlenecks and performance issues in workflow executions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class BottleneckDetector(BaseModel):
    """
    Detects bottlenecks and performance anomalies in workflow executions.
    """
    
    def __init__(self, detection_method: str = 'isolation_forest'):
        """
        Initialize Bottleneck Detector.
        
        Args:
            detection_method: Method to use ('isolation_forest', 'clustering', 'classification')
        """
        super().__init__(f"BottleneckDetector_{detection_method}")
        self.detection_method = detection_method
        self.scaler = StandardScaler()
        
        # Initialize detection model based on method
        if detection_method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_jobs=-1
            )
        elif detection_method == 'clustering':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        elif detection_method == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown detection method: {detection_method}")
            
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'BottleneckDetector':
        """
        Fit the bottleneck detector.
        
        Args:
            X: Feature matrix (step-level data)
            y: Target labels (optional, for supervised methods)
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            X_prepared = self._prepare_bottleneck_features(X)
            self.feature_columns = X_prepared.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_prepared)
            
            if self.detection_method == 'classification':
                if y is None:
                    # Create labels based on duration percentiles
                    y = self._create_bottleneck_labels(X)
                    
                self.model.fit(X_scaled, y)
                self._log_training_info(X_prepared, y)
                
                # Evaluate classification performance
                y_pred = self.model.predict(X_scaled)
                self._evaluate_classification(y, y_pred)
                
            else:
                # Unsupervised methods
                self.model.fit(X_scaled)
                self._log_training_info(X_prepared, pd.Series([0] * len(X_prepared)))
                
                if self.detection_method == 'isolation_forest':
                    # Evaluate anomaly detection
                    anomaly_scores = self.model.decision_function(X_scaled)
                    self._evaluate_anomaly_detection(X_prepared, anomaly_scores)
                elif self.detection_method == 'clustering':
                    # Evaluate clustering
                    cluster_labels = self.model.fit_predict(X_scaled)
                    self._evaluate_clustering(X_prepared, cluster_labels)
            
            self.is_fitted = True
            logger.info(f"{self.model_name} training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error training {self.model_name}: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Detect bottlenecks in workflow steps.
        
        Args:
            X: Feature matrix
            
        Returns:
            Bottleneck predictions (1 = bottleneck, 0 = normal)
        """
        try:
            X_prepared = self._prepare_bottleneck_features(X)
            X_validated = self.validate_input(pd.DataFrame(X_prepared, columns=self.feature_columns))
            X_scaled = self.scaler.transform(X_validated)
            
            if self.detection_method == 'isolation_forest':
                # Isolation forest returns -1 for anomalies, 1 for normal
                predictions = self.model.predict(X_scaled)
                return (predictions == -1).astype(int)  # Convert to 1 for bottlenecks
                
            elif self.detection_method == 'clustering':
                # DBSCAN: -1 indicates noise/outliers (potential bottlenecks)
                cluster_labels = self.model.fit_predict(X_scaled)
                return (cluster_labels == -1).astype(int)
                
            elif self.detection_method == 'classification':
                return self.model.predict(X_scaled)
                
        except Exception as e:
            logger.error(f"Error making predictions with {self.model_name}: {e}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict bottleneck probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array
        """
        try:
            X_prepared = self._prepare_bottleneck_features(X)
            X_validated = self.validate_input(pd.DataFrame(X_prepared, columns=self.feature_columns))
            X_scaled = self.scaler.transform(X_validated)
            
            if self.detection_method == 'isolation_forest':
                # Convert decision function scores to probabilities
                scores = self.model.decision_function(X_scaled)
                # Normalize scores to 0-1 range (higher score = more normal)
                proba_normal = (scores - scores.min()) / (scores.max() - scores.min())
                proba_bottleneck = 1 - proba_normal
                return np.column_stack([proba_normal, proba_bottleneck])
                
            elif self.detection_method == 'classification':
                return self.model.predict_proba(X_scaled)
                
            else:
                # For clustering, return binary probabilities
                predictions = self.predict(X)
                proba_normal = 1 - predictions
                proba_bottleneck = predictions
                return np.column_stack([proba_normal, proba_bottleneck])
                
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.model_name}: {e}")
            raise
            
    def get_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores for each step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores array
        """
        try:
            X_prepared = self._prepare_bottleneck_features(X)
            X_validated = self.validate_input(pd.DataFrame(X_prepared, columns=self.feature_columns))
            X_scaled = self.scaler.transform(X_validated)
            
            if self.detection_method == 'isolation_forest':
                return self.model.decision_function(X_scaled)
            elif self.detection_method == 'classification' and hasattr(self.model, 'predict_proba'):
                # Use probability of being a bottleneck as anomaly score
                proba = self.model.predict_proba(X_scaled)
                return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                # For clustering, use distance to nearest cluster center
                predictions = self.predict(X)
                return predictions.astype(float)
                
        except Exception as e:
            logger.error(f"Error getting anomaly scores with {self.model_name}: {e}")
            raise
            
    def analyze_bottlenecks(self, X: pd.DataFrame, steps_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive bottleneck analysis.
        
        Args:
            X: Feature matrix
            steps_df: Original steps DataFrame with metadata
            
        Returns:
            Dictionary with bottleneck analysis results
        """
        try:
            # Get predictions and scores
            predictions = self.predict(X)
            scores = self.get_anomaly_scores(X)
            
            # Identify bottleneck steps
            bottleneck_indices = np.where(predictions == 1)[0]
            
            results = {
                'total_steps': len(predictions),
                'bottleneck_count': len(bottleneck_indices),
                'bottleneck_rate': len(bottleneck_indices) / len(predictions),
                'bottleneck_steps': [],
                'step_type_analysis': {},
                'execution_analysis': {},
                'recommendations': []
            }
            
            if len(bottleneck_indices) > 0:
                # Analyze bottleneck steps
                bottleneck_steps_data = steps_df.iloc[bottleneck_indices]
                
                for idx in bottleneck_indices:
                    step_info = {
                        'step_id': steps_df.iloc[idx]['step_id'] if 'step_id' in steps_df.columns else idx,
                        'step_name': steps_df.iloc[idx]['step_name'] if 'step_name' in steps_df.columns else f'step_{idx}',
                        'step_type': steps_df.iloc[idx]['step_type'] if 'step_type' in steps_df.columns else 'unknown',
                        'duration': steps_df.iloc[idx]['duration'] if 'duration' in steps_df.columns else 0,
                        'anomaly_score': float(scores[idx]),
                        'execution_id': steps_df.iloc[idx]['execution_id'] if 'execution_id' in steps_df.columns else None
                    }
                    results['bottleneck_steps'].append(step_info)
                
                # Analyze by step type
                if 'step_type' in bottleneck_steps_data.columns:
                    step_type_counts = bottleneck_steps_data['step_type'].value_counts()
                    total_by_type = steps_df['step_type'].value_counts()
                    
                    for step_type, count in step_type_counts.items():
                        total_count = total_by_type.get(step_type, 0)
                        results['step_type_analysis'][step_type] = {
                            'bottleneck_count': int(count),
                            'total_count': int(total_count),
                            'bottleneck_rate': float(count / total_count) if total_count > 0 else 0
                        }
                
                # Analyze by execution
                if 'execution_id' in bottleneck_steps_data.columns:
                    execution_counts = bottleneck_steps_data['execution_id'].value_counts()
                    
                    for execution_id, count in execution_counts.items():
                        execution_steps = steps_df[steps_df['execution_id'] == execution_id]
                        results['execution_analysis'][str(execution_id)] = {
                            'bottleneck_count': int(count),
                            'total_steps': len(execution_steps),
                            'bottleneck_rate': float(count / len(execution_steps))
                        }
                
                # Generate recommendations
                results['recommendations'] = self._generate_recommendations(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing bottlenecks: {e}")
            raise
            
    def _prepare_bottleneck_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features specifically for bottleneck detection.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Prepared feature DataFrame
        """
        # Select relevant features for bottleneck detection
        bottleneck_features = []
        
        # Duration-related features
        duration_cols = [col for col in X.columns if 'duration' in col.lower()]
        bottleneck_features.extend(duration_cols)
        
        # Performance-related features
        performance_cols = [col for col in X.columns if any(keyword in col.lower() 
                          for keyword in ['performance', 'score', 'percentile', 'wait'])]
        bottleneck_features.extend(performance_cols)
        
        # Position-related features
        position_cols = [col for col in X.columns if any(keyword in col.lower() 
                        for keyword in ['position', 'step_count', 'is_first', 'is_last'])]
        bottleneck_features.extend(position_cols)
        
        # Error-related features
        error_cols = [col for col in X.columns if any(keyword in col.lower() 
                     for keyword in ['error', 'failed', 'anomaly'])]
        bottleneck_features.extend(error_cols)
        
        # Remove duplicates and select available columns
        bottleneck_features = list(set(bottleneck_features))
        available_features = [col for col in bottleneck_features if col in X.columns]
        
        if not available_features:
            # Fallback to all numeric columns
            available_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X_bottleneck = X[available_features].copy()
        
        # Handle missing values
        X_bottleneck = X_bottleneck.fillna(0)
        
        # Remove infinite values
        X_bottleneck = X_bottleneck.replace([np.inf, -np.inf], 0)
        
        return X_bottleneck
        
    def _create_bottleneck_labels(self, X: pd.DataFrame) -> pd.Series:
        """
        Create bottleneck labels based on duration percentiles.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Binary labels (1 = bottleneck, 0 = normal)
        """
        # Use duration-based labeling as default
        duration_col = None
        for col in X.columns:
            if 'duration' in col.lower() and 'percentile' not in col.lower():
                duration_col = col
                break
        
        if duration_col is None:
            # Create random labels as fallback
            logger.warning("No duration column found, creating random labels")
            return pd.Series(np.random.binomial(1, 0.1, len(X)))
        
        # Label top 10% duration as bottlenecks
        duration_threshold = X[duration_col].quantile(0.9)
        labels = (X[duration_col] > duration_threshold).astype(int)
        
        logger.info(f"Created labels: {labels.sum()} bottlenecks out of {len(labels)} steps")
        return labels
        
    def _evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            
            metrics = {
                'accuracy': float(report['accuracy']),
                'precision': float(report['1']['precision']) if '1' in report else 0,
                'recall': float(report['1']['recall']) if '1' in report else 0,
                'f1_score': float(report['1']['f1-score']) if '1' in report else 0,
                'confusion_matrix': cm.tolist()
            }
            
            self.model_metadata['performance_metrics'] = metrics
            logger.info(f"Classification accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.warning(f"Error evaluating classification: {e}")
            
    def _evaluate_anomaly_detection(self, X: pd.DataFrame, scores: np.ndarray):
        """
        Evaluate anomaly detection performance.
        
        Args:
            X: Feature DataFrame
            scores: Anomaly scores
        """
        try:
            metrics = {
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max()),
                'anomaly_threshold': float(np.percentile(scores, 10))  # Bottom 10% as anomalies
            }
            
            self.model_metadata['performance_metrics'] = metrics
            logger.info(f"Anomaly detection mean score: {metrics['mean_score']:.4f}")
            
        except Exception as e:
            logger.warning(f"Error evaluating anomaly detection: {e}")
            
    def _evaluate_clustering(self, X: pd.DataFrame, labels: np.ndarray):
        """
        Evaluate clustering performance.
        
        Args:
            X: Feature DataFrame
            labels: Cluster labels
        """
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # Exclude noise
            n_noise = list(labels).count(-1)
            
            metrics = {
                'n_clusters': int(n_clusters),
                'n_noise_points': int(n_noise),
                'noise_ratio': float(n_noise / len(labels)),
                'cluster_sizes': {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
            }
            
            self.model_metadata['performance_metrics'] = metrics
            logger.info(f"Clustering: {n_clusters} clusters, {n_noise} noise points")
            
        except Exception as e:
            logger.warning(f"Error evaluating clustering: {e}")
            
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on bottleneck analysis.
        
        Args:
            analysis_results: Bottleneck analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Overall bottleneck rate
        bottleneck_rate = analysis_results['bottleneck_rate']
        if bottleneck_rate > 0.15:
            recommendations.append(f"High bottleneck rate ({bottleneck_rate:.1%}). Consider workflow optimization.")
        
        # Step type analysis
        step_type_analysis = analysis_results.get('step_type_analysis', {})
        for step_type, data in step_type_analysis.items():
            if data['bottleneck_rate'] > 0.3:
                recommendations.append(f"Step type '{step_type}' has high bottleneck rate ({data['bottleneck_rate']:.1%}). Review implementation.")
        
        # Execution analysis
        execution_analysis = analysis_results.get('execution_analysis', {})
        problematic_executions = [exec_id for exec_id, data in execution_analysis.items() 
                                if data['bottleneck_rate'] > 0.5]
        
        if len(problematic_executions) > len(execution_analysis) * 0.2:
            recommendations.append("Multiple executions have high bottleneck rates. Consider system-wide optimization.")
        
        if not recommendations:
            recommendations.append("Bottleneck levels are within normal range. Continue monitoring.")
        
        return recommendations
