"""
Data Loader for FlowMind ML Pipeline

Handles loading workflow execution data from database and external sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and manages workflow execution data for ML pipeline.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize DataLoader with database connection.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = sa.create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def load_workflow_executions(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        workflow_ids: Optional[List[int]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load workflow execution data from database.
        
        Args:
            start_date: Filter executions after this date
            end_date: Filter executions before this date  
            workflow_ids: Filter by specific workflow IDs
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with workflow execution data
        """
        try:
            query = """
            SELECT 
                we.id as execution_id,
                we.workflow_id,
                w.name as workflow_name,
                we.status,
                we.start_time,
                we.end_time,
                we.duration,
                we.input_data,
                we.output_data,
                we.execution_path,
                we.bottlenecks,
                we.optimization_opportunities,
                w.avg_duration,
                w.success_rate,
                w.optimization_score
            FROM workflow_executions we
            JOIN workflows w ON we.workflow_id = w.id
            WHERE 1=1
            """
            
            params = {}
            
            if start_date:
                query += " AND we.start_time >= %(start_date)s"
                params['start_date'] = start_date
                
            if end_date:
                query += " AND we.start_time <= %(end_date)s"
                params['end_date'] = end_date
                
            if workflow_ids:
                query += " AND we.workflow_id IN %(workflow_ids)s"
                params['workflow_ids'] = tuple(workflow_ids)
                
            query += " ORDER BY we.start_time DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                
            df = pd.read_sql(query, self.engine, params=params)
            
            # Convert datetime columns
            datetime_cols = ['start_time', 'end_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    
            logger.info(f"Loaded {len(df)} workflow executions")
            return df
            
        except Exception as e:
            logger.error(f"Error loading workflow executions: {e}")
            raise
            
    def load_execution_steps(
        self,
        execution_ids: Optional[List[int]] = None,
        step_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load execution step data.
        
        Args:
            execution_ids: Filter by specific execution IDs
            step_types: Filter by step types
            
        Returns:
            DataFrame with execution step data
        """
        try:
            query = """
            SELECT 
                es.id as step_id,
                es.execution_id,
                es.step_name,
                es.step_type,
                es.status,
                es.start_time,
                es.end_time,
                es.duration,
                es.input_data,
                es.output_data,
                es.error_message,
                es.performance_score,
                es.anomaly_score
            FROM execution_steps es
            WHERE 1=1
            """
            
            params = {}
            
            if execution_ids:
                query += " AND es.execution_id IN %(execution_ids)s"
                params['execution_ids'] = tuple(execution_ids)
                
            if step_types:
                query += " AND es.step_type IN %(step_types)s"
                params['step_types'] = tuple(step_types)
                
            query += " ORDER BY es.execution_id, es.start_time"
            
            df = pd.read_sql(query, self.engine, params=params)
            
            # Convert datetime columns
            datetime_cols = ['start_time', 'end_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    
            logger.info(f"Loaded {len(df)} execution steps")
            return df
            
        except Exception as e:
            logger.error(f"Error loading execution steps: {e}")
            raise
            
    def load_optimizations(
        self,
        workflow_ids: Optional[List[int]] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load workflow optimization data.
        
        Args:
            workflow_ids: Filter by workflow IDs
            status: Filter by optimization status
            
        Returns:
            DataFrame with optimization data
        """
        try:
            query = """
            SELECT 
                wo.id as optimization_id,
                wo.workflow_id,
                wo.optimization_type,
                wo.suggestion,
                wo.impact_score,
                wo.confidence_score,
                wo.status,
                wo.created_at,
                wo.applied_at,
                wo.expected_improvement,
                wo.actual_improvement
            FROM workflow_optimizations wo
            WHERE 1=1
            """
            
            params = {}
            
            if workflow_ids:
                query += " AND wo.workflow_id IN %(workflow_ids)s"
                params['workflow_ids'] = tuple(workflow_ids)
                
            if status:
                query += " AND wo.status = %(status)s"
                params['status'] = status
                
            query += " ORDER BY wo.created_at DESC"
            
            df = pd.read_sql(query, self.engine, params=params)
            
            # Convert datetime columns
            datetime_cols = ['created_at', 'applied_at']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    
            logger.info(f"Loaded {len(df)} optimizations")
            return df
            
        except Exception as e:
            logger.error(f"Error loading optimizations: {e}")
            raise
            
    def create_training_dataset(
        self,
        lookback_days: int = 30,
        min_executions: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive training dataset for ML models.
        
        Args:
            lookback_days: Number of days to look back for data
            min_executions: Minimum executions required per workflow
            
        Returns:
            Dictionary containing different datasets
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Load main data
            executions_df = self.load_workflow_executions(start_date, end_date)
            
            if len(executions_df) == 0:
                logger.warning("No execution data found for training")
                return {}
            
            # Filter workflows with sufficient data
            workflow_counts = executions_df['workflow_id'].value_counts()
            valid_workflows = workflow_counts[workflow_counts >= min_executions].index.tolist()
            
            executions_df = executions_df[executions_df['workflow_id'].isin(valid_workflows)]
            
            # Load related data
            execution_ids = executions_df['execution_id'].tolist()
            steps_df = self.load_execution_steps(execution_ids)
            optimizations_df = self.load_optimizations(valid_workflows)
            
            return {
                'executions': executions_df,
                'steps': steps_df,
                'optimizations': optimizations_df,
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'num_workflows': len(valid_workflows),
                    'num_executions': len(executions_df),
                    'num_steps': len(steps_df)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            raise
            
    def save_dataset(self, dataset: Dict[str, pd.DataFrame], filepath: str):
        """
        Save dataset to file for later use.
        
        Args:
            dataset: Dataset dictionary
            filepath: Path to save dataset
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as pickle for now (could add other formats)
            pd.to_pickle(dataset, filepath)
            logger.info(f"Dataset saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
            
    def load_dataset(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """
        Load previously saved dataset.
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            Dataset dictionary
        """
        try:
            dataset = pd.read_pickle(filepath)
            logger.info(f"Dataset loaded from {filepath}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
