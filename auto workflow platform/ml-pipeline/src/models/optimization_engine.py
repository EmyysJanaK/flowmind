"""
Optimization Engine for FlowMind ML Pipeline

Generates optimization recommendations for workflow improvements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class OptimizationEngine(BaseModel):
    """
    Generates optimization recommendations for workflow improvements.
    """
    
    def __init__(self, optimization_type: str = 'performance'):
        """
        Initialize Optimization Engine.
        
        Args:
            optimization_type: Type of optimization ('performance', 'cost', 'reliability')
        """
        super().__init__(f"OptimizationEngine_{optimization_type}")
        self.optimization_type = optimization_type
        self.impact_model = None
        self.confidence_model = None
        self.recommendation_templates = self._load_recommendation_templates()
        
        # Initialize models
        self.impact_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        self.confidence_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'OptimizationEngine':
        """
        Fit the optimization engine.
        
        Args:
            X: Feature matrix (workflow/execution features)
            y: Target DataFrame with optimization outcomes
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            X_prepared = self._prepare_features(X)
            self.feature_columns = X_prepared.columns.tolist()
            
            # Prepare targets
            if 'impact_score' not in y.columns or 'confidence_score' not in y.columns:
                # Create synthetic targets if not provided
                y = self._create_optimization_targets(X_prepared, y)
            
            y_impact = y['impact_score']
            y_confidence = y['confidence_score']
            
            # Remove outliers
            impact_q95 = y_impact.quantile(0.95)
            valid_indices = (y_impact <= impact_q95) & (y_confidence >= 0) & (y_confidence <= 1)
            
            X_clean = X_prepared[valid_indices]
            y_impact_clean = y_impact[valid_indices]
            y_confidence_clean = y_confidence[valid_indices]
            
            # Fit models
            logger.info("Training impact prediction model...")
            self.impact_model.fit(X_clean, y_impact_clean)
            
            logger.info("Training confidence prediction model...")
            self.confidence_model.fit(X_clean, y_confidence_clean)
            
            self.is_fitted = True
            self._log_training_info(X_clean, y_impact_clean)
            
            # Evaluate models
            self._evaluate_models(X_clean, y_impact_clean, y_confidence_clean)
            
            logger.info(f"{self.model_name} training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error training {self.model_name}: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict optimization impact and confidence scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with 'impact_score' and 'confidence_score' predictions
        """
        try:
            X_validated = self.validate_input(X)
            
            impact_pred = self.impact_model.predict(X_validated)
            confidence_pred = self.confidence_model.predict(X_validated)
            
            # Ensure predictions are within reasonable bounds
            impact_pred = np.maximum(impact_pred, 0)
            confidence_pred = np.clip(confidence_pred, 0, 1)
            
            return {
                'impact_score': impact_pred,
                'confidence_score': confidence_pred
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with {self.model_name}: {e}")
            raise
            
    def generate_recommendations(
        self, 
        workflow_data: pd.DataFrame,
        execution_data: Optional[pd.DataFrame] = None,
        bottleneck_data: Optional[pd.DataFrame] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations for workflows.
        
        Args:
            workflow_data: Workflow-level data
            execution_data: Execution-level data (optional)
            bottleneck_data: Bottleneck analysis data (optional)
            top_k: Number of top recommendations to return
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = []
            
            for idx, workflow in workflow_data.iterrows():
                workflow_id = workflow.get('workflow_id', idx)
                
                # Analyze workflow performance
                workflow_recommendations = self._analyze_workflow_performance(
                    workflow, execution_data, bottleneck_data
                )
                
                # Predict impact and confidence for each recommendation
                for rec in workflow_recommendations:
                    rec['workflow_id'] = workflow_id
                    
                    # Create feature vector for prediction
                    feature_vector = self._create_recommendation_features(
                        workflow, rec, execution_data
                    )
                    
                    if len(feature_vector) > 0:
                        # Predict impact and confidence
                        predictions = self.predict(pd.DataFrame([feature_vector]))
                        rec['predicted_impact'] = float(predictions['impact_score'][0])
                        rec['predicted_confidence'] = float(predictions['confidence_score'][0])
                        rec['priority_score'] = rec['predicted_impact'] * rec['predicted_confidence']
                    else:
                        rec['predicted_impact'] = rec.get('impact_score', 0.5)
                        rec['predicted_confidence'] = rec.get('confidence_score', 0.5)
                        rec['priority_score'] = rec['predicted_impact'] * rec['predicted_confidence']
                
                recommendations.extend(workflow_recommendations)
            
            # Sort by priority score and return top k
            recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return recommendations[:top_k]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
            
    def optimize_workflow_parameters(
        self, 
        workflow_data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Optimize workflow parameters using the trained models.
        
        Args:
            workflow_data: Current workflow data
            parameter_ranges: Dictionary of parameter names and their ranges
            
        Returns:
            Dictionary with optimized parameters and expected improvement
        """
        try:
            if len(workflow_data) == 0:
                raise ValueError("No workflow data provided")
            
            current_features = self._prepare_features(workflow_data.iloc[[0]])
            current_predictions = self.predict(current_features)
            
            best_improvement = 0
            best_parameters = {}
            best_features = current_features.copy()
            
            # Grid search over parameter space
            n_trials = 100
            
            for trial in range(n_trials):
                # Generate random parameter values
                trial_parameters = {}
                trial_features = current_features.copy()
                
                for param_name, (min_val, max_val) in parameter_ranges.items():
                    param_value = np.random.uniform(min_val, max_val)
                    trial_parameters[param_name] = param_value
                    
                    # Update feature vector if parameter exists
                    if param_name in trial_features.columns:
                        trial_features[param_name] = param_value
                
                # Predict impact with new parameters
                trial_predictions = self.predict(trial_features)
                
                # Calculate improvement
                improvement = (trial_predictions['impact_score'][0] - 
                             current_predictions['impact_score'][0])
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_parameters = trial_parameters
                    best_features = trial_features
            
            result = {
                'optimized_parameters': best_parameters,
                'expected_improvement': float(best_improvement),
                'current_impact': float(current_predictions['impact_score'][0]),
                'optimized_impact': float(current_predictions['impact_score'][0] + best_improvement),
                'confidence': float(self.predict(best_features)['confidence_score'][0])
            }
            
            logger.info(f"Parameter optimization completed. Expected improvement: {best_improvement:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            raise
            
    def _analyze_workflow_performance(
        self, 
        workflow: pd.Series, 
        execution_data: Optional[pd.DataFrame] = None,
        bottleneck_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze workflow performance and generate specific recommendations.
        
        Args:
            workflow: Single workflow data
            execution_data: Execution-level data
            bottleneck_data: Bottleneck analysis data
            
        Returns:
            List of recommendations for this workflow
        """
        recommendations = []
        
        # Performance-based recommendations
        if self.optimization_type == 'performance':
            # Duration optimization
            avg_duration = workflow.get('avg_duration', 0)
            if avg_duration > 300:  # 5 minutes
                recommendations.append({
                    'type': 'duration_optimization',
                    'description': 'Optimize workflow duration by parallelizing steps',
                    'impact_score': min(avg_duration / 1800, 1.0),  # Normalize by 30 minutes
                    'confidence_score': 0.8,
                    'category': 'performance',
                    'implementation_effort': 'medium'
                })
            
            # Success rate optimization
            success_rate = workflow.get('success_rate', 1.0)
            if success_rate < 0.9:
                recommendations.append({
                    'type': 'reliability_improvement',
                    'description': 'Add error handling and retry mechanisms',
                    'impact_score': 1.0 - success_rate,
                    'confidence_score': 0.9,
                    'category': 'reliability',
                    'implementation_effort': 'low'
                })
        
        # Bottleneck-based recommendations
        if bottleneck_data is not None:
            workflow_id = workflow.get('workflow_id')
            workflow_bottlenecks = bottleneck_data[
                bottleneck_data.get('workflow_id') == workflow_id
            ] if 'workflow_id' in bottleneck_data.columns else pd.DataFrame()
            
            if len(workflow_bottlenecks) > 0:
                bottleneck_rate = workflow_bottlenecks['is_bottleneck'].mean() if 'is_bottleneck' in workflow_bottlenecks.columns else 0
                
                if bottleneck_rate > 0.1:
                    recommendations.append({
                        'type': 'bottleneck_resolution',
                        'description': f'Address bottlenecks in {bottleneck_rate:.1%} of steps',
                        'impact_score': bottleneck_rate,
                        'confidence_score': 0.7,
                        'category': 'performance',
                        'implementation_effort': 'high'
                    })
        
        # Resource optimization
        if execution_data is not None:
            workflow_id = workflow.get('workflow_id')
            workflow_executions = execution_data[
                execution_data.get('workflow_id') == workflow_id
            ] if 'workflow_id' in execution_data.columns else pd.DataFrame()
            
            if len(workflow_executions) > 0:
                # Check for resource usage patterns
                duration_variance = workflow_executions['duration'].std() if 'duration' in workflow_executions.columns else 0
                mean_duration = workflow_executions['duration'].mean() if 'duration' in workflow_executions.columns else 0
                
                if mean_duration > 0 and duration_variance / mean_duration > 0.5:
                    recommendations.append({
                        'type': 'resource_optimization',
                        'description': 'Optimize resource allocation for consistent performance',
                        'impact_score': min(duration_variance / mean_duration, 1.0),
                        'confidence_score': 0.6,
                        'category': 'resource',
                        'implementation_effort': 'medium'
                    })
        
        # General optimization recommendations
        if len(recommendations) == 0:
            recommendations.append({
                'type': 'monitoring_enhancement',
                'description': 'Enhance monitoring and logging for better insights',
                'impact_score': 0.3,
                'confidence_score': 0.9,
                'category': 'monitoring',
                'implementation_effort': 'low'
            })
        
        return recommendations
        
    def _create_optimization_targets(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic optimization targets when not provided.
        
        Args:
            X: Feature matrix
            y: Original target data
            
        Returns:
            DataFrame with impact_score and confidence_score
        """
        targets = pd.DataFrame(index=X.index)
        
        # Create impact scores based on duration and success rate
        if 'duration' in y.columns:
            # Higher duration -> higher potential impact
            duration_normalized = (y['duration'] - y['duration'].min()) / (y['duration'].max() - y['duration'].min())
            targets['impact_score'] = duration_normalized * 0.7
        else:
            targets['impact_score'] = np.random.uniform(0.1, 0.8, len(X))
        
        if 'success_rate' in y.columns:
            # Lower success rate -> higher potential impact
            targets['impact_score'] += (1 - y['success_rate']) * 0.3
        
        # Create confidence scores based on data quality
        targets['confidence_score'] = np.random.uniform(0.5, 0.9, len(X))
        
        # Adjust confidence based on data completeness
        missing_ratio = X.isnull().sum(axis=1) / len(X.columns)
        targets['confidence_score'] = targets['confidence_score'] * (1 - missing_ratio * 0.5)
        
        # Ensure bounds
        targets['impact_score'] = np.clip(targets['impact_score'], 0, 1)
        targets['confidence_score'] = np.clip(targets['confidence_score'], 0, 1)
        
        return targets
        
    def _create_recommendation_features(
        self, 
        workflow: pd.Series, 
        recommendation: Dict[str, Any],
        execution_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Create features for recommendation impact prediction.
        
        Args:
            workflow: Workflow data
            recommendation: Recommendation data
            execution_data: Execution data (optional)
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Workflow features
        features.update({
            'avg_duration': workflow.get('avg_duration', 0),
            'success_rate': workflow.get('success_rate', 1.0),
            'optimization_score': workflow.get('optimization_score', 0.5),
            'total_executions': workflow.get('total_executions', 0)
        })
        
        # Recommendation features
        features.update({
            'recommendation_impact': recommendation.get('impact_score', 0.5),
            'recommendation_confidence': recommendation.get('confidence_score', 0.5),
            'implementation_effort_score': self._encode_effort(
                recommendation.get('implementation_effort', 'medium')
            )
        })
        
        # Category encoding
        category_scores = {
            'performance': 0.8,
            'reliability': 0.9,
            'resource': 0.6,
            'monitoring': 0.3
        }
        features['category_score'] = category_scores.get(
            recommendation.get('category', 'monitoring'), 0.5
        )
        
        return features
        
    def _encode_effort(self, effort: str) -> float:
        """
        Encode implementation effort as numeric score.
        
        Args:
            effort: Effort level string
            
        Returns:
            Numeric effort score
        """
        effort_map = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }
        return effort_map.get(effort.lower(), 0.5)
        
    def _load_recommendation_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load recommendation templates.
        
        Returns:
            Dictionary of recommendation templates
        """
        return {
            'duration_optimization': {
                'description_template': 'Optimize workflow duration by {action}',
                'actions': ['parallelizing steps', 'caching results', 'optimizing queries'],
                'typical_impact': 0.6,
                'typical_confidence': 0.8
            },
            'reliability_improvement': {
                'description_template': 'Improve reliability by {action}',
                'actions': ['adding retry logic', 'error handling', 'input validation'],
                'typical_impact': 0.4,
                'typical_confidence': 0.9
            },
            'resource_optimization': {
                'description_template': 'Optimize resources by {action}',
                'actions': ['scaling resources', 'load balancing', 'memory optimization'],
                'typical_impact': 0.5,
                'typical_confidence': 0.6
            }
        }
        
    def _evaluate_models(self, X: pd.DataFrame, y_impact: pd.Series, y_confidence: pd.Series):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y_impact: Impact target
            y_confidence: Confidence target
        """
        try:
            # Make predictions
            impact_pred = self.impact_model.predict(X)
            confidence_pred = self.confidence_model.predict(X)
            
            # Calculate metrics
            impact_metrics = {
                'mae': float(mean_absolute_error(y_impact, impact_pred)),
                'r2': float(r2_score(y_impact, impact_pred))
            }
            
            confidence_metrics = {
                'mae': float(mean_absolute_error(y_confidence, confidence_pred)),
                'r2': float(r2_score(y_confidence, confidence_pred))
            }
            
            self.model_metadata['performance_metrics'] = {
                'impact_model': impact_metrics,
                'confidence_model': confidence_metrics
            }
            
            logger.info(f"Impact model R² score: {impact_metrics['r2']:.4f}")
            logger.info(f"Confidence model R² score: {confidence_metrics['r2']:.4f}")
            
        except Exception as e:
            logger.warning(f"Error evaluating models: {e}")
