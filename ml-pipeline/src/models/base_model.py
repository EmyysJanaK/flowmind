"""
Base Model class for FlowMind ML Pipeline

Provides common functionality for all ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all FlowMind ML models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.feature_columns = []
        self.model_metadata = {
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'version': '1.0.0',
            'training_data_shape': None,
            'feature_columns': [],
            'performance_metrics': {}
        }
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model doesn't support probability prediction")
            
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_scores = np.abs(self.model.coef_).flatten()
        else:
            logger.warning("Model doesn't provide feature importance")
            return {}
            
        feature_importance = dict(zip(self.feature_columns, importance_scores))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
    def validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input data.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Validated feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Check if required columns are present
        missing_columns = set(self.feature_columns) - set(X.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Select and order columns correctly
        X_validated = X[self.feature_columns].copy()
        
        # Handle missing values
        X_validated = X_validated.fillna(0)
        
        return X_validated
        
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model data
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_metadata': self.model_metadata,
                'is_fitted': self.is_fitted
            }
            
            # Save using joblib
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            # Restore model state
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.model_metadata = model_data['model_metadata']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'metadata': self.model_metadata
        }
        
        if self.is_fitted and hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
            
        return info
        
    def update_metadata(self, key: str, value: Any):
        """
        Update model metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.model_metadata[key] = value
        self.model_metadata['updated_at'] = datetime.now().isoformat()
        
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Prepared feature matrix
        """
        # Remove non-numeric columns and handle missing values
        X_prepared = X.select_dtypes(include=[np.number]).copy()
        
        # Fill missing values
        X_prepared = X_prepared.fillna(0)
        
        # Remove infinite values
        X_prepared = X_prepared.replace([np.inf, -np.inf], 0)
        
        return X_prepared
        
    def _log_training_info(self, X: pd.DataFrame, y: pd.Series):
        """
        Log information about training data.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        self.model_metadata.update({
            'training_data_shape': X.shape,
            'feature_columns': X.columns.tolist(),
            'target_distribution': y.value_counts().to_dict() if hasattr(y, 'value_counts') else {'mean': float(y.mean()), 'std': float(y.std())},
            'last_trained': datetime.now().isoformat()
        })
        
        logger.info(f"Training {self.model_name} with {X.shape[0]} samples and {X.shape[1]} features")
