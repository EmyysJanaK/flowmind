"""
Model Metrics and Evaluation Utilities for FlowMind ML Pipeline

Provides comprehensive evaluation metrics and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelMetrics:
    """
    Comprehensive model evaluation and metrics calculation.
    """
    
    def __init__(self):
        """Initialize ModelMetrics."""
        self.evaluation_history = []
        
    def evaluate_regression(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "regression_model"
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with regression metrics
        """
        try:
            metrics = {
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred)),
                'mape': float(self._calculate_mape(y_true, y_pred)),
                'max_error': float(np.max(np.abs(y_true - y_pred))),
                'mean_residual': float(np.mean(y_true - y_pred)),
                'std_residual': float(np.std(y_true - y_pred))
            }
            
            # Add custom metrics
            metrics.update(self._calculate_custom_regression_metrics(y_true, y_pred))
            
            # Store evaluation
            self._store_evaluation(model_name, 'regression', metrics)
            
            logger.info(f"Regression evaluation for {model_name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            raise
            
    def evaluate_classification(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "classification_model",
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Evaluate classification model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model being evaluated
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary with classification metrics
        """
        try:
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
            # ROC AUC for binary or multi-class with probabilities
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                        else:
                            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
                    else:
                        # Multi-class classification
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average=average))
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
                    metrics['roc_auc'] = None
            
            # Add custom metrics
            metrics.update(self._calculate_custom_classification_metrics(y_true, y_pred, y_proba))
            
            # Store evaluation
            self._store_evaluation(model_name, 'classification', metrics)
            
            logger.info(f"Classification evaluation for {model_name}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            raise
            
    def evaluate_anomaly_detection(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        threshold: Optional[float] = None,
        model_name: str = "anomaly_model"
    ) -> Dict[str, float]:
        """
        Evaluate anomaly detection model performance.
        
        Args:
            y_true: True binary labels (1 = anomaly, 0 = normal)
            y_scores: Anomaly scores
            threshold: Threshold for converting scores to predictions
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with anomaly detection metrics
        """
        try:
            # Determine threshold if not provided
            if threshold is None:
                threshold = np.percentile(y_scores, 90)  # Top 10% as anomalies
            
            # Convert scores to predictions
            y_pred = (y_scores > threshold).astype(int)
            
            # Basic classification metrics
            metrics = self.evaluate_classification(y_true, y_pred, model_name=f"{model_name}_classification")
            
            # Anomaly-specific metrics
            try:
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
                metrics['pr_auc'] = float(np.trapz(precision, recall))
                
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
                metrics['roc_auc'] = float(np.trapz(tpr, fpr))
                
            except Exception as e:
                logger.warning(f"Could not calculate curve metrics: {e}")
                metrics['pr_auc'] = None
                metrics['roc_auc'] = None
            
            # Detection rate at different percentiles
            for percentile in [95, 90, 85]:
                thresh = np.percentile(y_scores, percentile)
                pred = (y_scores > thresh).astype(int)
                detection_rate = recall_score(y_true, pred, zero_division=0)
                metrics[f'detection_rate_p{percentile}'] = float(detection_rate)
            
            metrics['threshold_used'] = float(threshold)
            
            # Store evaluation
            self._store_evaluation(model_name, 'anomaly_detection', metrics)
            
            logger.info(f"Anomaly detection evaluation for {model_name}: PR-AUC = {metrics.get('pr_auc', 'N/A')}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating anomaly detection model: {e}")
            raise
            
    def evaluate_time_series_prediction(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        model_name: str = "timeseries_model"
    ) -> Dict[str, float]:
        """
        Evaluate time series prediction model performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Timestamps for the predictions (optional)
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with time series metrics
        """
        try:
            # Basic regression metrics
            metrics = self.evaluate_regression(y_true, y_pred, f"{model_name}_regression")
            
            # Time series specific metrics
            residuals = y_true - y_pred
            
            # Directional accuracy
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                directional_accuracy = np.mean(true_direction == pred_direction)
                metrics['directional_accuracy'] = float(directional_accuracy)
            
            # Autocorrelation of residuals
            if len(residuals) > 1:
                autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                metrics['residual_autocorrelation'] = float(autocorr) if not np.isnan(autocorr) else 0.0
            
            # Trend accuracy
            if len(y_true) > 2:
                true_trend = np.polyfit(range(len(y_true)), y_true, 1)[0]
                pred_trend = np.polyfit(range(len(y_pred)), y_pred, 1)[0]
                trend_error = abs(true_trend - pred_trend)
                metrics['trend_error'] = float(trend_error)
            
            # Store evaluation
            self._store_evaluation(model_name, 'time_series', metrics)
            
            logger.info(f"Time series evaluation for {model_name}: Directional accuracy = {metrics.get('directional_accuracy', 'N/A')}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating time series model: {e}")
            raise
            
    def compare_models(
        self, 
        model_results: Dict[str, Dict[str, float]],
        metric: str = 'r2'
    ) -> pd.DataFrame:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            model_results: Dictionary with model names and their metrics
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with model comparison
        """
        try:
            comparison_data = []
            
            for model_name, metrics in model_results.items():
                if metric in metrics:
                    comparison_data.append({
                        'model': model_name,
                        'metric_value': metrics[metric],
                        'rank': 0  # Will be filled later
                    })
            
            if not comparison_data:
                logger.warning(f"No models found with metric '{metric}'")
                return pd.DataFrame()
            
            # Create DataFrame and rank models
            df = pd.DataFrame(comparison_data)
            
            # Rank based on metric (higher is better for most metrics)
            ascending = metric.lower() in ['mae', 'mse', 'rmse', 'mape']
            df['rank'] = df['metric_value'].rank(ascending=ascending, method='dense').astype(int)
            
            # Sort by rank
            df = df.sort_values('rank').reset_index(drop=True)
            
            logger.info(f"Model comparison completed. Best model: {df.iloc[0]['model']}")
            return df
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
            
    def calculate_feature_importance_metrics(
        self, 
        feature_importance: Dict[str, float],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate metrics for feature importance analysis.
        
        Args:
            feature_importance: Dictionary with feature names and importance scores
            top_k: Number of top features to analyze
            
        Returns:
            Dictionary with feature importance metrics
        """
        try:
            if not feature_importance:
                return {}
            
            importance_values = np.array(list(feature_importance.values()))
            feature_names = list(feature_importance.keys())
            
            # Sort by importance
            sorted_indices = np.argsort(importance_values)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_importance = importance_values[sorted_indices]
            
            metrics = {
                'total_features': len(feature_importance),
                'top_features': sorted_features[:top_k],
                'top_importance_scores': sorted_importance[:top_k].tolist(),
                'importance_concentration': float(sorted_importance[:5].sum() / sorted_importance.sum()),
                'importance_entropy': float(self._calculate_entropy(importance_values)),
                'importance_gini': float(self._calculate_gini_coefficient(importance_values))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating feature importance metrics: {e}")
            raise
            
    def generate_performance_report(
        self, 
        model_name: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report for a model.
        
        Args:
            model_name: Name of the model
            include_history: Whether to include evaluation history
            
        Returns:
            Dictionary with performance report
        """
        try:
            # Find evaluations for this model
            model_evaluations = [
                eval_data for eval_data in self.evaluation_history 
                if eval_data['model_name'] == model_name
            ]
            
            if not model_evaluations:
                logger.warning(f"No evaluations found for model '{model_name}'")
                return {}
            
            latest_evaluation = model_evaluations[-1]
            
            report = {
                'model_name': model_name,
                'evaluation_type': latest_evaluation['evaluation_type'],
                'latest_metrics': latest_evaluation['metrics'],
                'evaluation_timestamp': latest_evaluation['timestamp'],
                'total_evaluations': len(model_evaluations)
            }
            
            if include_history and len(model_evaluations) > 1:
                # Calculate performance trends
                metric_trends = self._calculate_metric_trends(model_evaluations)
                report['metric_trends'] = metric_trends
                
                # Performance summary
                report['performance_summary'] = self._generate_performance_summary(
                    latest_evaluation['metrics'], 
                    latest_evaluation['evaluation_type']
                )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise
            
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
    def _calculate_custom_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate custom regression metrics."""
        metrics = {}
        
        # Median Absolute Error
        metrics['median_ae'] = float(np.median(np.abs(y_true - y_pred)))
        
        # Explained Variance
        metrics['explained_variance'] = float(1 - np.var(y_true - y_pred) / np.var(y_true))
        
        # Coefficient of Determination (alternative calculation)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2_manual'] = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        return metrics
        
    def _calculate_custom_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate custom classification metrics."""
        metrics = {}
        
        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        
        # Matthews Correlation Coefficient
        from sklearn.metrics import matthews_corrcoef
        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
        
        # Class distribution
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique_classes.astype(str), class_counts.astype(int)))
        
        return metrics
        
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate entropy of values."""
        # Normalize values to probabilities
        probs = values / np.sum(values)
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log2(probs))
        
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
    def _store_evaluation(
        self, 
        model_name: str, 
        evaluation_type: str, 
        metrics: Dict[str, Any]
    ):
        """Store evaluation results in history."""
        evaluation_data = {
            'model_name': model_name,
            'evaluation_type': evaluation_type,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.evaluation_history.append(evaluation_data)
        
    def _calculate_metric_trends(
        self, 
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate trends in metrics over time."""
        trends = {}
        
        # Group metrics by type
        metric_series = {}
        for eval_data in evaluations:
            for metric_name, metric_value in eval_data['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    if metric_name not in metric_series:
                        metric_series[metric_name] = []
                    metric_series[metric_name].append(metric_value)
        
        # Calculate trends
        for metric_name, values in metric_series.items():
            if len(values) > 1:
                # Linear trend
                x = np.arange(len(values))
                slope = np.corrcoef(x, values)[0, 1] if len(values) > 2 else 0
                
                trends[metric_name] = {
                    'latest_value': float(values[-1]),
                    'trend_slope': float(slope),
                    'improvement': float(values[-1] - values[0]),
                    'best_value': float(max(values) if metric_name not in ['mae', 'mse', 'rmse'] else min(values))
                }
        
        return trends
        
    def _generate_performance_summary(
        self, 
        metrics: Dict[str, Any], 
        evaluation_type: str
    ) -> str:
        """Generate human-readable performance summary."""
        if evaluation_type == 'regression':
            r2 = metrics.get('r2', 0)
            rmse = metrics.get('rmse', 0)
            
            if r2 > 0.9:
                performance = "Excellent"
            elif r2 > 0.7:
                performance = "Good"
            elif r2 > 0.5:
                performance = "Fair"
            else:
                performance = "Poor"
                
            return f"{performance} regression performance (R² = {r2:.3f}, RMSE = {rmse:.3f})"
            
        elif evaluation_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            
            if accuracy > 0.9 and f1 > 0.9:
                performance = "Excellent"
            elif accuracy > 0.8 and f1 > 0.8:
                performance = "Good"
            elif accuracy > 0.7 and f1 > 0.7:
                performance = "Fair"
            else:
                performance = "Poor"
                
            return f"{performance} classification performance (Accuracy = {accuracy:.3f}, F1 = {f1:.3f})"
            
        else:
            return "Performance metrics available"
