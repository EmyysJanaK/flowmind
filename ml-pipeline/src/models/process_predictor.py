"""
Process Predictor for FlowMind ML Pipeline

Predicts workflow execution duration and success probability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ProcessPredictor(BaseModel):
    """
    Predicts workflow execution duration and performance metrics.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize Process Predictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'linear')
        """
        super().__init__(f"ProcessPredictor_{model_type}")
        self.model_type = model_type
        self.duration_model = None
        self.success_model = None
        
        # Initialize models based on type
        if model_type == 'random_forest':
            self.duration_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.success_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.duration_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.success_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'linear':
            self.duration_model = LinearRegression()
            self.success_model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> 'ProcessPredictor':
        """
        Fit the process predictor models.
        
        Args:
            X: Feature matrix
            y: Target DataFrame with 'duration' and 'success_rate' columns
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            X_prepared = self._prepare_features(X)
            self.feature_columns = X_prepared.columns.tolist()
            
            # Prepare targets
            if 'duration' not in y.columns:
                raise ValueError("Target DataFrame must contain 'duration' column")
                
            # Create success rate if not provided
            if 'success_rate' not in y.columns:
                if 'status' in y.columns:
                    y['success_rate'] = (y['status'] == 'completed').astype(float)
                else:
                    logger.warning("No success_rate or status column found, using default value of 1.0")
                    y['success_rate'] = 1.0
            
            # Remove outliers from duration
            duration_q99 = y['duration'].quantile(0.99)
            valid_indices = y['duration'] <= duration_q99
            
            X_clean = X_prepared[valid_indices]
            y_duration = y.loc[valid_indices, 'duration']
            y_success = y.loc[valid_indices, 'success_rate']
            
            # Fit duration model
            logger.info("Training duration prediction model...")
            self.duration_model.fit(X_clean, y_duration)
            
            # Fit success rate model
            logger.info("Training success rate prediction model...")
            self.success_model.fit(X_clean, y_success)
            
            self.is_fitted = True
            self._log_training_info(X_clean, y_duration)
            
            # Calculate and store performance metrics
            self._evaluate_models(X_clean, y_duration, y_success)
            
            logger.info(f"{self.model_name} training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error training {self.model_name}: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict duration and success rate for workflow executions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with 'duration' and 'success_rate' predictions
        """
        try:
            X_validated = self.validate_input(X)
            
            # Make predictions
            duration_pred = self.duration_model.predict(X_validated)
            success_pred = self.success_model.predict(X_validated)
            
            # Ensure predictions are within reasonable bounds
            duration_pred = np.maximum(duration_pred, 0)  # Duration can't be negative
            success_pred = np.clip(success_pred, 0, 1)    # Success rate between 0 and 1
            
            return {
                'duration': duration_pred,
                'success_rate': success_pred
            }
            
        except Exception as e:
            logger.error(f"Error making predictions with {self.model_name}: {e}")
            raise
            
    def predict_duration(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict only execution duration.
        
        Args:
            X: Feature matrix
            
        Returns:
            Duration predictions array
        """
        predictions = self.predict(X)
        return predictions['duration']
        
    def predict_success_rate(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict only success rate.
        
        Args:
            X: Feature matrix
            
        Returns:
            Success rate predictions array
        """
        predictions = self.predict(X)
        return predictions['success_rate']
        
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict with confidence intervals (for ensemble models).
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        try:
            X_validated = self.validate_input(X)
            
            results = {}
            
            # Duration predictions with confidence
            if hasattr(self.duration_model, 'estimators_'):
                # For ensemble models, calculate prediction variance
                duration_preds = np.array([
                    estimator.predict(X_validated) 
                    for estimator in self.duration_model.estimators_
                ])
                
                results['duration'] = {
                    'prediction': duration_preds.mean(axis=0),
                    'std': duration_preds.std(axis=0),
                    'confidence_lower': np.percentile(duration_preds, 5, axis=0),
                    'confidence_upper': np.percentile(duration_preds, 95, axis=0)
                }
            else:
                # For non-ensemble models, just return predictions
                duration_pred = self.duration_model.predict(X_validated)
                results['duration'] = {
                    'prediction': duration_pred,
                    'std': np.zeros_like(duration_pred),
                    'confidence_lower': duration_pred,
                    'confidence_upper': duration_pred
                }
            
            # Success rate predictions with confidence
            if hasattr(self.success_model, 'estimators_'):
                success_preds = np.array([
                    estimator.predict(X_validated) 
                    for estimator in self.success_model.estimators_
                ])
                
                results['success_rate'] = {
                    'prediction': np.clip(success_preds.mean(axis=0), 0, 1),
                    'std': success_preds.std(axis=0),
                    'confidence_lower': np.clip(np.percentile(success_preds, 5, axis=0), 0, 1),
                    'confidence_upper': np.clip(np.percentile(success_preds, 95, axis=0), 0, 1)
                }
            else:
                success_pred = np.clip(self.success_model.predict(X_validated), 0, 1)
                results['success_rate'] = {
                    'prediction': success_pred,
                    'std': np.zeros_like(success_pred),
                    'confidence_lower': success_pred,
                    'confidence_upper': success_pred
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making confidence predictions with {self.model_name}: {e}")
            raise
            
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance for both duration and success rate models.
        
        Returns:
            Dictionary with feature importance for each model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        importance = {}
        
        # Duration model importance
        if hasattr(self.duration_model, 'feature_importances_'):
            duration_importance = dict(zip(
                self.feature_columns, 
                self.duration_model.feature_importances_
            ))
            importance['duration'] = dict(sorted(
                duration_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        # Success rate model importance
        if hasattr(self.success_model, 'feature_importances_'):
            success_importance = dict(zip(
                self.feature_columns, 
                self.success_model.feature_importances_
            ))
            importance['success_rate'] = dict(sorted(
                success_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        return importance
        
    def _evaluate_models(self, X: pd.DataFrame, y_duration: pd.Series, y_success: pd.Series):
        """
        Evaluate model performance and store metrics.
        
        Args:
            X: Feature matrix
            y_duration: Duration targets
            y_success: Success rate targets
        """
        try:
            # Split data for evaluation
            X_train, X_test, y_dur_train, y_dur_test, y_suc_train, y_suc_test = train_test_split(
                X, y_duration, y_success, test_size=0.2, random_state=42
            )
            
            # Duration model metrics
            dur_pred = self.duration_model.predict(X_test)
            duration_metrics = {
                'mae': float(mean_absolute_error(y_dur_test, dur_pred)),
                'mse': float(mean_squared_error(y_dur_test, dur_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_dur_test, dur_pred))),
                'r2': float(r2_score(y_dur_test, dur_pred))
            }
            
            # Success rate model metrics
            suc_pred = self.success_model.predict(X_test)
            success_metrics = {
                'mae': float(mean_absolute_error(y_suc_test, suc_pred)),
                'mse': float(mean_squared_error(y_suc_test, suc_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_suc_test, suc_pred))),
                'r2': float(r2_score(y_suc_test, suc_pred))
            }
            
            # Store metrics
            self.model_metadata['performance_metrics'] = {
                'duration_model': duration_metrics,
                'success_model': success_metrics
            }
            
            logger.info(f"Duration model R² score: {duration_metrics['r2']:.4f}")
            logger.info(f"Success model R² score: {success_metrics['r2']:.4f}")
            
        except Exception as e:
            logger.warning(f"Error evaluating models: {e}")
            
    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on the models.
        
        Args:
            X: Feature matrix
            y: Target DataFrame
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        try:
            X_prepared = self._prepare_features(X)
            
            # Prepare targets
            y_duration = y['duration']
            y_success = y.get('success_rate', pd.Series([1.0] * len(y)))
            
            # Remove outliers
            duration_q99 = y_duration.quantile(0.99)
            valid_indices = y_duration <= duration_q99
            
            X_clean = X_prepared[valid_indices]
            y_dur_clean = y_duration[valid_indices]
            y_suc_clean = y_success[valid_indices]
            
            # Cross-validate duration model
            duration_scores = cross_val_score(
                self.duration_model, X_clean, y_dur_clean, 
                cv=cv, scoring='r2'
            )
            
            # Cross-validate success model
            success_scores = cross_val_score(
                self.success_model, X_clean, y_suc_clean, 
                cv=cv, scoring='r2'
            )
            
            results = {
                'duration_model': {
                    'mean_score': float(duration_scores.mean()),
                    'std_score': float(duration_scores.std()),
                    'scores': duration_scores.tolist()
                },
                'success_model': {
                    'mean_score': float(success_scores.mean()),
                    'std_score': float(success_scores.std()),
                    'scores': success_scores.tolist()
                }
            }
            
            logger.info(f"Duration model CV score: {results['duration_model']['mean_score']:.4f} (+/- {results['duration_model']['std_score']:.4f})")
            logger.info(f"Success model CV score: {results['success_model']['mean_score']:.4f} (+/- {results['success_model']['std_score']:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
