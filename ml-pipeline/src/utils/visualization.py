"""
Visualization Utilities for FlowMind ML Pipeline

Provides comprehensive visualization capabilities for model analysis and results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization utilities for ML pipeline analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize Visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_model_performance_comparison(
        self, 
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of model performance across multiple metrics.
        
        Args:
            model_results: Dictionary with model names and their metrics
            metrics: List of metrics to compare
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            if metrics is None:
                # Auto-detect common metrics
                all_metrics = set()
                for model_metrics in model_results.values():
                    all_metrics.update(model_metrics.keys())
                
                common_metrics = ['r2', 'accuracy', 'f1_score', 'mae', 'rmse']
                metrics = [m for m in common_metrics if m in all_metrics]
                
                if not metrics:
                    metrics = list(all_metrics)[:5]  # Take first 5 metrics
            
            n_metrics = len(metrics)
            n_models = len(model_results)
            
            if n_metrics == 0 or n_models == 0:
                logger.warning("No metrics or models to plot")
                return plt.figure()
            
            # Create subplots
            fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                model_names = []
                metric_values = []
                
                for model_name, model_metrics in model_results.items():
                    if metric in model_metrics:
                        model_names.append(model_name)
                        metric_values.append(model_metrics[metric])
                
                if metric_values:
                    # Create bar plot
                    bars = axes[i].bar(model_names, metric_values, 
                                     color=self.color_palette[:len(model_names)])
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, metric_values):
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom')
                    
                    axes[i].set_title(f'{metric.upper()}')
                    axes[i].set_ylabel('Score')
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Highlight best performing model
                    best_idx = np.argmax(metric_values) if metric not in ['mae', 'mse', 'rmse'] else np.argmin(metric_values)
                    bars[best_idx].set_color('gold')
                    bars[best_idx].set_edgecolor('black')
                    bars[best_idx].set_linewidth(2)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Model comparison plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting model performance comparison: {e}")
            return plt.figure()
            
    def plot_feature_importance(
        self, 
        feature_importance: Dict[str, float],
        top_k: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary with feature names and importance scores
            top_k: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            if not feature_importance:
                logger.warning("No feature importance data to plot")
                return plt.figure()
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_k]
            
            features, importance_scores = zip(*top_features)
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
            
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importance_scores, color=self.color_palette[0])
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Top feature at the top
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top {len(features)} Feature Importance')
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center', fontsize=9)
            
            # Highlight top 3 features
            for i in range(min(3, len(bars))):
                bars[i].set_color('gold')
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            return plt.figure()
            
    def plot_prediction_vs_actual(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot predictions vs actual values for regression models.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot: Predicted vs Actual
            ax1.scatter(y_true, y_pred, alpha=0.6, color=self.color_palette[0])
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate and display metrics
            r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'{model_name} - Predictions vs Actual\nR² = {r2:.3f}, MAE = {mae:.3f}, RMSE = {rmse:.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residual plot
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, color=self.color_palette[1])
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title(f'{model_name} - Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Prediction vs actual plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting predictions vs actual: {e}")
            return plt.figure()
            
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix for classification models.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.metrics import confusion_matrix, accuracy_score
            
            cm = confusion_matrix(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(cm.shape[0])]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{model_name} - Confusion Matrix\nAccuracy = {accuracy:.3f}')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            return plt.figure()
            
    def plot_workflow_performance_timeline(
        self, 
        workflow_data: pd.DataFrame,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot workflow performance metrics over time.
        
        Args:
            workflow_data: DataFrame with workflow execution data
            metrics: List of metrics to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            if metrics is None:
                metrics = ['duration', 'success_rate']
            
            # Ensure we have time column
            time_col = None
            for col in ['start_time', 'created_at', 'timestamp']:
                if col in workflow_data.columns:
                    time_col = col
                    break
            
            if time_col is None:
                logger.warning("No time column found in workflow data")
                return plt.figure()
            
            # Convert time column to datetime
            workflow_data = workflow_data.copy()
            workflow_data[time_col] = pd.to_datetime(workflow_data[time_col])
            
            # Sort by time
            workflow_data = workflow_data.sort_values(time_col)
            
            n_metrics = len(metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                if metric in workflow_data.columns:
                    # Plot time series
                    axes[i].plot(workflow_data[time_col], workflow_data[metric], 
                               marker='o', linewidth=2, markersize=4, 
                               color=self.color_palette[i % len(self.color_palette)])
                    
                    # Add trend line
                    if len(workflow_data) > 2:
                        z = np.polyfit(range(len(workflow_data)), workflow_data[metric], 1)
                        p = np.poly1d(z)
                        axes[i].plot(workflow_data[time_col], p(range(len(workflow_data))), 
                                   "r--", alpha=0.7, linewidth=2, label='Trend')
                    
                    axes[i].set_xlabel('Time')
                    axes[i].set_ylabel(metric.replace('_', ' ').title())
                    axes[i].set_title(f'Workflow {metric.replace("_", " ").title()} Over Time')
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend()
                    
                    # Format x-axis
                    fig.autofmt_xdate()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Workflow timeline plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting workflow performance timeline: {e}")
            return plt.figure()
            
    def plot_bottleneck_analysis(
        self, 
        bottleneck_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bottleneck analysis results.
        
        Args:
            bottleneck_data: DataFrame with bottleneck analysis data
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Bottleneck distribution by step type
            if 'step_type' in bottleneck_data.columns and 'is_bottleneck' in bottleneck_data.columns:
                step_type_analysis = bottleneck_data.groupby('step_type')['is_bottleneck'].agg(['count', 'sum'])
                step_type_analysis['bottleneck_rate'] = step_type_analysis['sum'] / step_type_analysis['count']
                
                axes[0, 0].bar(step_type_analysis.index, step_type_analysis['bottleneck_rate'],
                             color=self.color_palette[0])
                axes[0, 0].set_title('Bottleneck Rate by Step Type')
                axes[0, 0].set_ylabel('Bottleneck Rate')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Duration distribution for bottlenecks vs normal steps
            if 'duration' in bottleneck_data.columns and 'is_bottleneck' in bottleneck_data.columns:
                bottleneck_durations = bottleneck_data[bottleneck_data['is_bottleneck'] == 1]['duration']
                normal_durations = bottleneck_data[bottleneck_data['is_bottleneck'] == 0]['duration']
                
                axes[0, 1].hist(normal_durations, bins=30, alpha=0.7, label='Normal Steps', 
                              color=self.color_palette[1])
                axes[0, 1].hist(bottleneck_durations, bins=30, alpha=0.7, label='Bottlenecks', 
                              color=self.color_palette[2])
                axes[0, 1].set_title('Duration Distribution')
                axes[0, 1].set_xlabel('Duration (seconds)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
            
            # 3. Bottlenecks over time
            if 'start_time' in bottleneck_data.columns:
                bottleneck_data_copy = bottleneck_data.copy()
                bottleneck_data_copy['start_time'] = pd.to_datetime(bottleneck_data_copy['start_time'])
                bottleneck_data_copy['date'] = bottleneck_data_copy['start_time'].dt.date
                
                daily_bottlenecks = bottleneck_data_copy.groupby('date')['is_bottleneck'].agg(['count', 'sum'])
                daily_bottlenecks['bottleneck_rate'] = daily_bottlenecks['sum'] / daily_bottlenecks['count']
                
                axes[1, 0].plot(daily_bottlenecks.index, daily_bottlenecks['bottleneck_rate'], 
                              marker='o', color=self.color_palette[3])
                axes[1, 0].set_title('Daily Bottleneck Rate')
                axes[1, 0].set_ylabel('Bottleneck Rate')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Anomaly score distribution
            if 'anomaly_score' in bottleneck_data.columns:
                axes[1, 1].hist(bottleneck_data['anomaly_score'], bins=30, color=self.color_palette[4])
                axes[1, 1].set_title('Anomaly Score Distribution')
                axes[1, 1].set_xlabel('Anomaly Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Bottleneck analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting bottleneck analysis: {e}")
            return plt.figure()
            
    def plot_optimization_recommendations(
        self, 
        recommendations: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot optimization recommendations analysis.
        
        Args:
            recommendations: List of recommendation dictionaries
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            if not recommendations:
                logger.warning("No recommendations to plot")
                return plt.figure()
            
            # Convert to DataFrame for easier analysis
            rec_df = pd.DataFrame(recommendations)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Recommendations by type
            if 'type' in rec_df.columns:
                type_counts = rec_df['type'].value_counts()
                axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                             colors=self.color_palette[:len(type_counts)])
                axes[0, 0].set_title('Recommendations by Type')
            
            # 2. Impact vs Confidence scatter
            if 'predicted_impact' in rec_df.columns and 'predicted_confidence' in rec_df.columns:
                scatter = axes[0, 1].scatter(rec_df['predicted_confidence'], rec_df['predicted_impact'],
                                          c=rec_df.get('priority_score', 1), cmap='viridis', 
                                          s=100, alpha=0.7)
                axes[0, 1].set_xlabel('Confidence')
                axes[0, 1].set_ylabel('Impact')
                axes[0, 1].set_title('Impact vs Confidence')
                plt.colorbar(scatter, ax=axes[0, 1], label='Priority Score')
            
            # 3. Priority scores distribution
            if 'priority_score' in rec_df.columns:
                axes[1, 0].hist(rec_df['priority_score'], bins=20, color=self.color_palette[0])
                axes[1, 0].set_xlabel('Priority Score')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Priority Score Distribution')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Implementation effort distribution
            if 'implementation_effort' in rec_df.columns:
                effort_counts = rec_df['implementation_effort'].value_counts()
                axes[1, 1].bar(effort_counts.index, effort_counts.values, color=self.color_palette[1])
                axes[1, 1].set_xlabel('Implementation Effort')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Implementation Effort Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Optimization recommendations plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting optimization recommendations: {e}")
            return plt.figure()
            
    def create_dashboard(
        self, 
        model_results: Dict[str, Dict[str, float]],
        workflow_data: pd.DataFrame,
        bottleneck_data: Optional[pd.DataFrame] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with all visualizations.
        
        Args:
            model_results: Model performance results
            workflow_data: Workflow execution data
            bottleneck_data: Bottleneck analysis data (optional)
            recommendations: Optimization recommendations (optional)
            save_path: Path to save the dashboard
            
        Returns:
            Matplotlib figure
        """
        try:
            fig = plt.figure(figsize=(20, 24))
            
            # Create grid layout
            gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
            
            # 1. Model performance comparison (top row)
            ax1 = fig.add_subplot(gs[0, :2])
            if model_results:
                # Simplified model comparison
                model_names = list(model_results.keys())
                r2_scores = [results.get('r2', results.get('accuracy', 0)) for results in model_results.values()]
                
                bars = ax1.bar(model_names, r2_scores, color=self.color_palette[:len(model_names)])
                ax1.set_title('Model Performance Comparison')
                ax1.set_ylabel('Performance Score')
                ax1.tick_params(axis='x', rotation=45)
                
                # Highlight best model
                if r2_scores:
                    best_idx = np.argmax(r2_scores)
                    bars[best_idx].set_color('gold')
            
            # 2. Workflow performance timeline (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            if 'start_time' in workflow_data.columns and 'duration' in workflow_data.columns:
                workflow_data_copy = workflow_data.copy()
                workflow_data_copy['start_time'] = pd.to_datetime(workflow_data_copy['start_time'])
                workflow_data_sorted = workflow_data_copy.sort_values('start_time')
                
                ax2.plot(workflow_data_sorted['start_time'], workflow_data_sorted['duration'], 
                        marker='o', markersize=3, linewidth=1)
                ax2.set_title('Workflow Duration Timeline')
                ax2.set_ylabel('Duration (seconds)')
                fig.autofmt_xdate()
            
            # 3-6. Additional plots based on available data
            row_idx = 1
            
            # Bottleneck analysis
            if bottleneck_data is not None and len(bottleneck_data) > 0:
                ax3 = fig.add_subplot(gs[row_idx, :2])
                if 'step_type' in bottleneck_data.columns and 'is_bottleneck' in bottleneck_data.columns:
                    step_analysis = bottleneck_data.groupby('step_type')['is_bottleneck'].mean()
                    ax3.bar(step_analysis.index, step_analysis.values, color=self.color_palette[2])
                    ax3.set_title('Bottleneck Rate by Step Type')
                    ax3.set_ylabel('Bottleneck Rate')
                    ax3.tick_params(axis='x', rotation=45)
                
                ax4 = fig.add_subplot(gs[row_idx, 2:])
                if 'duration' in bottleneck_data.columns:
                    ax4.hist(bottleneck_data['duration'], bins=30, color=self.color_palette[3])
                    ax4.set_title('Step Duration Distribution')
                    ax4.set_xlabel('Duration (seconds)')
                    ax4.set_ylabel('Frequency')
                
                row_idx += 1
            
            # Recommendations analysis
            if recommendations and len(recommendations) > 0:
                rec_df = pd.DataFrame(recommendations)
                
                ax5 = fig.add_subplot(gs[row_idx, :2])
                if 'type' in rec_df.columns:
                    type_counts = rec_df['type'].value_counts()
                    ax5.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
                    ax5.set_title('Recommendations by Type')
                
                ax6 = fig.add_subplot(gs[row_idx, 2:])
                if 'priority_score' in rec_df.columns:
                    ax6.hist(rec_df['priority_score'], bins=15, color=self.color_palette[4])
                    ax6.set_title('Recommendation Priority Distribution')
                    ax6.set_xlabel('Priority Score')
                    ax6.set_ylabel('Count')
                
                row_idx += 1
            
            # Add summary statistics text
            summary_ax = fig.add_subplot(gs[row_idx:, :])
            summary_ax.axis('off')
            
            summary_text = self._generate_dashboard_summary(
                model_results, workflow_data, bottleneck_data, recommendations
            )
            
            summary_ax.text(0.05, 0.95, summary_text, transform=summary_ax.transAxes,
                          fontsize=12, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.suptitle('FlowMind ML Pipeline Dashboard', fontsize=16, fontweight='bold')
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return plt.figure()
            
    def _generate_dashboard_summary(
        self, 
        model_results: Dict[str, Dict[str, float]],
        workflow_data: pd.DataFrame,
        bottleneck_data: Optional[pd.DataFrame] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate summary text for dashboard."""
        summary_lines = ["FLOWMIND ML PIPELINE SUMMARY", "=" * 40, ""]
        
        # Model performance summary
        if model_results:
            summary_lines.append("Model Performance:")
            for model_name, metrics in model_results.items():
                main_metric = metrics.get('r2', metrics.get('accuracy', 0))
                summary_lines.append(f"  • {model_name}: {main_metric:.3f}")
            summary_lines.append("")
        
        # Workflow summary
        summary_lines.append("Workflow Statistics:")
        summary_lines.append(f"  • Total executions: {len(workflow_data)}")
        
        if 'duration' in workflow_data.columns:
            avg_duration = workflow_data['duration'].mean()
            summary_lines.append(f"  • Average duration: {avg_duration:.1f}s")
        
        if 'success_rate' in workflow_data.columns:
            avg_success = workflow_data['success_rate'].mean()
            summary_lines.append(f"  • Average success rate: {avg_success:.1%}")
        
        summary_lines.append("")
        
        # Bottleneck summary
        if bottleneck_data is not None and len(bottleneck_data) > 0:
            summary_lines.append("Bottleneck Analysis:")
            if 'is_bottleneck' in bottleneck_data.columns:
                bottleneck_rate = bottleneck_data['is_bottleneck'].mean()
                summary_lines.append(f"  • Bottleneck rate: {bottleneck_rate:.1%}")
            
            if 'step_type' in bottleneck_data.columns:
                problematic_types = bottleneck_data.groupby('step_type')['is_bottleneck'].mean().head(3)
                summary_lines.append("  • Top problematic step types:")
                for step_type, rate in problematic_types.items():
                    summary_lines.append(f"    - {step_type}: {rate:.1%}")
            
            summary_lines.append("")
        
        # Recommendations summary
        if recommendations:
            summary_lines.append("Optimization Recommendations:")
            summary_lines.append(f"  • Total recommendations: {len(recommendations)}")
            
            rec_df = pd.DataFrame(recommendations)
            if 'type' in rec_df.columns:
                top_types = rec_df['type'].value_counts().head(3)
                summary_lines.append("  • Top recommendation types:")
                for rec_type, count in top_types.items():
                    summary_lines.append(f"    - {rec_type}: {count}")
        
        return "\n".join(summary_lines)
