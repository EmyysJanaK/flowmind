"""
Data Preprocessor for FlowMind ML Pipeline

Handles data cleaning, transformation, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocesses workflow execution data for ML models.
    """
    
    def __init__(self):
        """Initialize preprocessor with default configurations."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_columns = {}
        
    def clean_execution_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean workflow execution data.
        
        Args:
            df: Raw execution DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Remove rows with missing critical fields
            critical_fields = ['execution_id', 'workflow_id', 'status', 'start_time']
            df_clean = df_clean.dropna(subset=critical_fields)
            
            # Handle duration calculation
            if 'end_time' in df_clean.columns and 'start_time' in df_clean.columns:
                # Calculate duration if missing
                mask = df_clean['duration'].isna() & df_clean['end_time'].notna()
                if mask.any():
                    time_diff = (df_clean.loc[mask, 'end_time'] - 
                               df_clean.loc[mask, 'start_time'])
                    df_clean.loc[mask, 'duration'] = time_diff.dt.total_seconds()
            
            # Handle JSON fields safely
            json_fields = ['input_data', 'output_data', 'execution_path', 
                          'bottlenecks', 'optimization_opportunities']
            
            for field in json_fields:
                if field in df_clean.columns:
                    df_clean[field] = df_clean[field].apply(self._safe_json_parse)
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates(subset=['execution_id'])
            
            logger.info(f"Cleaned execution data: {len(df)} -> {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning execution data: {e}")
            raise
            
    def clean_step_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean execution step data.
        
        Args:
            df: Raw step DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df_clean = df.copy()
            
            # Remove rows with missing critical fields
            critical_fields = ['step_id', 'execution_id', 'step_name', 'step_type']
            df_clean = df_clean.dropna(subset=critical_fields)
            
            # Handle duration calculation for steps
            if 'end_time' in df_clean.columns and 'start_time' in df_clean.columns:
                mask = df_clean['duration'].isna() & df_clean['end_time'].notna()
                if mask.any():
                    time_diff = (df_clean.loc[mask, 'end_time'] - 
                               df_clean.loc[mask, 'start_time'])
                    df_clean.loc[mask, 'duration'] = time_diff.dt.total_seconds()
            
            # Handle JSON fields
            json_fields = ['input_data', 'output_data']
            for field in json_fields:
                if field in df_clean.columns:
                    df_clean[field] = df_clean[field].apply(self._safe_json_parse)
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates(subset=['step_id'])
            
            logger.info(f"Cleaned step data: {len(df)} -> {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning step data: {e}")
            raise
            
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from datetime columns.
        
        Args:
            df: DataFrame with datetime columns
            
        Returns:
            DataFrame with additional temporal features
        """
        try:
            df_temporal = df.copy()
            
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            for col in datetime_cols:
                base_name = col.replace('_time', '').replace('_at', '')
                
                # Extract basic temporal features
                df_temporal[f'{base_name}_hour'] = df[col].dt.hour
                df_temporal[f'{base_name}_day_of_week'] = df[col].dt.dayofweek
                df_temporal[f'{base_name}_month'] = df[col].dt.month
                df_temporal[f'{base_name}_quarter'] = df[col].dt.quarter
                df_temporal[f'{base_name}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
                
                # Business hours feature (9-17)
                df_temporal[f'{base_name}_is_business_hours'] = (
                    (df[col].dt.hour >= 9) & (df[col].dt.hour < 17)
                ).astype(int)
            
            logger.info(f"Extracted temporal features from {len(datetime_cols)} datetime columns")
            return df_temporal
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            raise
            
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        categorical_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with categorical columns
            categorical_cols: List of categorical column names
            fit: Whether to fit encoders (training) or use existing (inference)
            
        Returns:
            DataFrame with encoded features
        """
        try:
            df_encoded = df.copy()
            
            for col in categorical_cols:
                if col not in df.columns:
                    continue
                    
                if fit:
                    # Fit new encoder
                    encoder = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = encoder.fit_transform(
                        df[col].fillna('unknown')
                    )
                    self.encoders[col] = encoder
                else:
                    # Use existing encoder
                    if col in self.encoders:
                        encoder = self.encoders[col]
                        # Handle unseen categories
                        values = df[col].fillna('unknown')
                        encoded_values = []
                        
                        for val in values:
                            if val in encoder.classes_:
                                encoded_values.append(encoder.transform([val])[0])
                            else:
                                # Use a default value for unseen categories
                                encoded_values.append(-1)
                                
                        df_encoded[f'{col}_encoded'] = encoded_values
                    else:
                        logger.warning(f"No encoder found for column {col}")
            
            logger.info(f"Encoded {len(categorical_cols)} categorical features")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {e}")
            raise
            
    def scale_numerical_features(
        self, 
        df: pd.DataFrame, 
        numerical_cols: List[str],
        scaler_type: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame with numerical columns
            numerical_cols: List of numerical column names
            scaler_type: Type of scaler ('standard' or 'minmax')
            fit: Whether to fit scaler (training) or use existing (inference)
            
        Returns:
            DataFrame with scaled features
        """
        try:
            df_scaled = df.copy()
            
            if scaler_type == 'standard':
                scaler_class = StandardScaler
            elif scaler_type == 'minmax':
                scaler_class = MinMaxScaler
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
            
            for col in numerical_cols:
                if col not in df.columns:
                    continue
                    
                if fit:
                    # Fit new scaler
                    scaler = scaler_class()
                    values = df[col].values.reshape(-1, 1)
                    df_scaled[f'{col}_scaled'] = scaler.fit_transform(values).flatten()
                    self.scalers[col] = scaler
                else:
                    # Use existing scaler
                    if col in self.scalers:
                        scaler = self.scalers[col]
                        values = df[col].values.reshape(-1, 1)
                        df_scaled[f'{col}_scaled'] = scaler.transform(values).flatten()
                    else:
                        logger.warning(f"No scaler found for column {col}")
            
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
            return df_scaled
            
        except Exception as e:
            logger.error(f"Error scaling numerical features: {e}")
            raise
            
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: Dict[str, str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset.
        
        Args:
            df: DataFrame with missing values
            strategy: Dictionary mapping column types to imputation strategies
            fit: Whether to fit imputers (training) or use existing (inference)
            
        Returns:
            DataFrame with imputed values
        """
        try:
            if strategy is None:
                strategy = {
                    'numerical': 'median',
                    'categorical': 'most_frequent'
                }
                
            df_imputed = df.copy()
            
            # Get numerical and categorical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Handle numerical columns
            if 'numerical' in strategy and numerical_cols:
                if fit:
                    imputer = SimpleImputer(strategy=strategy['numerical'])
                    df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                    self.imputers['numerical'] = imputer
                else:
                    if 'numerical' in self.imputers:
                        imputer = self.imputers['numerical']
                        df_imputed[numerical_cols] = imputer.transform(df[numerical_cols])
            
            # Handle categorical columns
            if 'categorical' in strategy and categorical_cols:
                if fit:
                    imputer = SimpleImputer(strategy=strategy['categorical'])
                    df_imputed[categorical_cols] = imputer.fit_transform(df[categorical_cols])
                    self.imputers['categorical'] = imputer
                else:
                    if 'categorical' in self.imputers:
                        imputer = self.imputers['categorical']
                        df_imputed[categorical_cols] = imputer.transform(df[categorical_cols])
            
            logger.info("Handled missing values")
            return df_imputed
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
            
    def prepare_for_training(
        self, 
        executions_df: pd.DataFrame,
        steps_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare complete dataset for ML model training.
        
        Args:
            executions_df: Workflow executions DataFrame
            steps_df: Execution steps DataFrame (optional)
            
        Returns:
            Dictionary with prepared datasets
        """
        try:
            # Clean execution data
            exec_clean = self.clean_execution_data(executions_df)
            
            # Extract temporal features
            exec_temporal = self.extract_temporal_features(exec_clean)
            
            # Define feature types
            categorical_cols = ['status', 'workflow_name']
            numerical_cols = ['duration', 'avg_duration', 'success_rate', 'optimization_score']
            
            # Handle missing values
            exec_imputed = self.handle_missing_values(exec_temporal, fit=True)
            
            # Encode categorical features
            exec_encoded = self.encode_categorical_features(
                exec_imputed, categorical_cols, fit=True
            )
            
            # Scale numerical features
            exec_scaled = self.scale_numerical_features(
                exec_encoded, numerical_cols, fit=True
            )
            
            result = {'executions': exec_scaled}
            
            # Process steps if provided
            if steps_df is not None and len(steps_df) > 0:
                steps_clean = self.clean_step_data(steps_df)
                steps_temporal = self.extract_temporal_features(steps_clean)
                
                step_categorical_cols = ['step_type', 'status']
                step_numerical_cols = ['duration', 'performance_score', 'anomaly_score']
                
                steps_imputed = self.handle_missing_values(steps_temporal, fit=True)
                steps_encoded = self.encode_categorical_features(
                    steps_imputed, step_categorical_cols, fit=True
                )
                steps_scaled = self.scale_numerical_features(
                    steps_encoded, step_numerical_cols, fit=True
                )
                
                result['steps'] = steps_scaled
            
            logger.info("Prepared datasets for training")
            return result
            
        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            raise
            
    def prepare_for_inference(
        self, 
        executions_df: pd.DataFrame,
        steps_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for model inference using existing preprocessors.
        
        Args:
            executions_df: Workflow executions DataFrame
            steps_df: Execution steps DataFrame (optional)
            
        Returns:
            Dictionary with prepared datasets
        """
        try:
            # Clean execution data
            exec_clean = self.clean_execution_data(executions_df)
            
            # Extract temporal features
            exec_temporal = self.extract_temporal_features(exec_clean)
            
            # Define feature types
            categorical_cols = ['status', 'workflow_name']
            numerical_cols = ['duration', 'avg_duration', 'success_rate', 'optimization_score']
            
            # Handle missing values using existing imputers
            exec_imputed = self.handle_missing_values(exec_temporal, fit=False)
            
            # Encode categorical features using existing encoders
            exec_encoded = self.encode_categorical_features(
                exec_imputed, categorical_cols, fit=False
            )
            
            # Scale numerical features using existing scalers
            exec_scaled = self.scale_numerical_features(
                exec_encoded, numerical_cols, fit=False
            )
            
            result = {'executions': exec_scaled}
            
            # Process steps if provided
            if steps_df is not None and len(steps_df) > 0:
                steps_clean = self.clean_step_data(steps_df)
                steps_temporal = self.extract_temporal_features(steps_clean)
                
                step_categorical_cols = ['step_type', 'status']
                step_numerical_cols = ['duration', 'performance_score', 'anomaly_score']
                
                steps_imputed = self.handle_missing_values(steps_temporal, fit=False)
                steps_encoded = self.encode_categorical_features(
                    steps_imputed, step_categorical_cols, fit=False
                )
                steps_scaled = self.scale_numerical_features(
                    steps_encoded, step_numerical_cols, fit=False
                )
                
                result['steps'] = steps_scaled
            
            logger.info("Prepared datasets for inference")
            return result
            
        except Exception as e:
            logger.error(f"Error preparing data for inference: {e}")
            raise
            
    def _safe_json_parse(self, value: Any) -> Any:
        """
        Safely parse JSON values.
        
        Args:
            value: Value to parse
            
        Returns:
            Parsed value or original value if parsing fails
        """
        if pd.isna(value) or value is None:
            return {}
            
        if isinstance(value, (dict, list)):
            return value
            
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return {}
                
        return value
        
    def save_preprocessors(self, filepath: str):
        """
        Save fitted preprocessors to file.
        
        Args:
            filepath: Path to save preprocessors
        """
        try:
            import joblib
            preprocessors = {
                'scalers': self.scalers,
                'encoders': self.encoders,
                'imputers': self.imputers,
                'feature_columns': self.feature_columns
            }
            joblib.dump(preprocessors, filepath)
            logger.info(f"Preprocessors saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {e}")
            raise
            
    def load_preprocessors(self, filepath: str):
        """
        Load preprocessors from file.
        
        Args:
            filepath: Path to load preprocessors from
        """
        try:
            import joblib
            preprocessors = joblib.load(filepath)
            
            self.scalers = preprocessors.get('scalers', {})
            self.encoders = preprocessors.get('encoders', {})
            self.imputers = preprocessors.get('imputers', {})
            self.feature_columns = preprocessors.get('feature_columns', {})
            
            logger.info(f"Preprocessors loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {e}")
            raise
