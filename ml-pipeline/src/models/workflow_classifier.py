"""
Workflow Classifier for FlowMind ML Pipeline

Classifies workflows into categories and predicts their characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class WorkflowClassifier(BaseModel):
    """
    Classifies workflows into categories and predicts their characteristics.
    """
    
    def __init__(self, classification_type: str = 'complexity'):
        """
        Initialize Workflow Classifier.
        
        Args:
            classification_type: Type of classification ('complexity', 'pattern', 'performance')
        """
        super().__init__(f"WorkflowClassifier_{classification_type}")
        self.classification_type = classification_type
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Initialize classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'WorkflowClassifier':
        """
        Fit the workflow classifier.
        
        Args:
            X: Feature matrix (workflow-level features)
            y: Target labels (optional, will be created if not provided)
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        try:
            # Prepare features
            X_prepared = self._prepare_features(X)
            self.feature_columns = X_prepared.columns.tolist()
            
            # Create or prepare labels
            if y is None:
                y = self._create_classification_labels(X_prepared)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_.tolist()
            
            # Remove any rows with missing target values
            valid_indices = ~pd.isna(y)
            X_clean = X_prepared[valid_indices]
            y_clean = y_encoded[valid_indices]
            
            # Fit the model
            logger.info(f"Training {self.classification_type} classifier...")
            self.model.fit(X_clean, y_clean)
            
            self.is_fitted = True
            self._log_training_info(X_clean, pd.Series(y_clean))
            
            # Evaluate model
            self._evaluate_classifier(X_clean, y_clean)
            
            logger.info(f"{self.model_name} training completed successfully")
            logger.info(f"Classes: {self.class_names}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training {self.model_name}: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict workflow classes.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        try:
            X_validated = self.validate_input(X)
            predictions_encoded = self.model.predict(X_validated)
            
            # Decode predictions back to original labels
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with {self.model_name}: {e}")
            raise
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probability matrix
        """
        try:
            X_validated = self.validate_input(X)
            return self.model.predict_proba(X_validated)
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.model_name}: {e}")
            raise
            
    def classify_workflows(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Classify workflows with detailed results.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with classification results
        """
        try:
            predictions = self.predict(X)
            probabilities = self.predict_proba(X)
            
            results = pd.DataFrame({
                'workflow_id': X.index if 'workflow_id' not in X.columns else X['workflow_id'],
                'predicted_class': predictions,
                'confidence': probabilities.max(axis=1)
            })
            
            # Add probability for each class
            for i, class_name in enumerate(self.class_names):
                results[f'prob_{class_name}'] = probabilities[:, i]
            
            # Add classification insights
            results['classification_type'] = self.classification_type
            results['insights'] = results.apply(
                lambda row: self._generate_classification_insights(row), axis=1
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying workflows: {e}")
            raise
            
    def cluster_workflows(
        self, 
        X: pd.DataFrame, 
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster workflows into groups.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results
        """
        try:
            X_prepared = self._prepare_features(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_prepared)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(X_prepared, cluster_labels, kmeans)
            
            results = {
                'cluster_labels': cluster_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'cluster_analysis': cluster_analysis,
                'n_clusters': n_clusters,
                'inertia': float(kmeans.inertia_)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error clustering workflows: {e}")
            raise
            
    def get_workflow_recommendations(
        self, 
        workflow_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on workflow classification.
        
        Args:
            workflow_data: Workflow feature data
            
        Returns:
            List of recommendations
        """
        try:
            # Classify workflows
            classification_results = self.classify_workflows(workflow_data)
            
            recommendations = []
            
            for idx, row in classification_results.iterrows():
                workflow_id = row['workflow_id']
                predicted_class = row['predicted_class']
                confidence = row['confidence']
                
                # Generate class-specific recommendations
                class_recommendations = self._get_class_recommendations(
                    predicted_class, confidence, workflow_data.loc[idx] if idx in workflow_data.index else None
                )
                
                for rec in class_recommendations:
                    rec['workflow_id'] = workflow_id
                    rec['classification'] = predicted_class
                    rec['classification_confidence'] = confidence
                    
                recommendations.extend(class_recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating workflow recommendations: {e}")
            raise
            
    def _create_classification_labels(self, X: pd.DataFrame) -> pd.Series:
        """
        Create classification labels based on workflow characteristics.
        
        Args:
            X: Feature matrix
            
        Returns:
            Classification labels
        """
        labels = []
        
        for idx, row in X.iterrows():
            if self.classification_type == 'complexity':
                # Classify based on duration and step count
                avg_duration = row.get('avg_duration', 0)
                step_count = row.get('total_steps', row.get('step_count', 1))
                
                if avg_duration < 60 and step_count <= 3:
                    label = 'simple'
                elif avg_duration < 300 and step_count <= 7:
                    label = 'moderate'
                elif avg_duration < 1800 and step_count <= 15:
                    label = 'complex'
                else:
                    label = 'very_complex'
                    
            elif self.classification_type == 'pattern':
                # Classify based on execution patterns
                execution_frequency = row.get('executions_per_day', 0)
                success_rate = row.get('success_rate', 1.0)
                
                if execution_frequency > 10 and success_rate > 0.95:
                    label = 'high_frequency_stable'
                elif execution_frequency > 5 and success_rate > 0.9:
                    label = 'regular_stable'
                elif execution_frequency < 1 and success_rate > 0.8:
                    label = 'occasional_reliable'
                elif success_rate < 0.7:
                    label = 'unreliable'
                else:
                    label = 'irregular'
                    
            elif self.classification_type == 'performance':
                # Classify based on performance characteristics
                duration_cv = row.get('duration_cv', 0)
                optimization_score = row.get('optimization_score', 0.5)
                
                if duration_cv < 0.2 and optimization_score > 0.8:
                    label = 'high_performance'
                elif duration_cv < 0.5 and optimization_score > 0.6:
                    label = 'good_performance'
                elif duration_cv < 1.0 and optimization_score > 0.4:
                    label = 'fair_performance'
                else:
                    label = 'poor_performance'
                    
            else:
                label = 'unknown'
                
            labels.append(label)
        
        return pd.Series(labels, index=X.index)
        
    def _analyze_clusters(
        self, 
        X: pd.DataFrame, 
        cluster_labels: np.ndarray, 
        kmeans: KMeans
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze cluster characteristics.
        
        Args:
            X: Feature matrix
            cluster_labels: Cluster assignments
            kmeans: Fitted KMeans model
            
        Returns:
            Dictionary with cluster analysis
        """
        cluster_analysis = {}
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
                
            analysis = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'center': kmeans.cluster_centers_[cluster_id].tolist(),
                'characteristics': {},
                'representative_features': {}
            }
            
            # Analyze feature characteristics
            for col in X.columns:
                cluster_mean = cluster_data[col].mean()
                overall_mean = X[col].mean()
                
                analysis['characteristics'][col] = {
                    'mean': float(cluster_mean),
                    'std': float(cluster_data[col].std()),
                    'vs_overall': float(cluster_mean - overall_mean)
                }
                
                # Identify representative features (high deviation from overall mean)
                if abs(cluster_mean - overall_mean) > X[col].std():
                    analysis['representative_features'][col] = float(cluster_mean)
            
            # Generate cluster description
            analysis['description'] = self._generate_cluster_description(analysis)
            
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
        
    def _generate_cluster_description(self, analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable cluster description.
        
        Args:
            analysis: Cluster analysis data
            
        Returns:
            Cluster description string
        """
        size = analysis['size']
        percentage = analysis['percentage']
        rep_features = analysis['representative_features']
        
        description = f"Cluster with {size} workflows ({percentage:.1f}%)"
        
        if rep_features:
            # Find most distinctive features
            feature_descriptions = []
            
            for feature, value in rep_features.items():
                if 'duration' in feature.lower():
                    if value > 600:
                        feature_descriptions.append("long-running")
                    elif value < 60:
                        feature_descriptions.append("quick")
                elif 'success' in feature.lower():
                    if value > 0.9:
                        feature_descriptions.append("highly reliable")
                    elif value < 0.7:
                        feature_descriptions.append("unreliable")
                elif 'frequency' in feature.lower():
                    if value > 5:
                        feature_descriptions.append("frequently executed")
                    elif value < 1:
                        feature_descriptions.append("rarely executed")
            
            if feature_descriptions:
                description += f" - characterized by: {', '.join(feature_descriptions)}"
        
        return description
        
    def _generate_classification_insights(self, row: pd.Series) -> str:
        """
        Generate insights for a classification result.
        
        Args:
            row: Classification result row
            
        Returns:
            Insight string
        """
        predicted_class = row['predicted_class']
        confidence = row['confidence']
        
        insights = []
        
        # Confidence-based insights
        if confidence > 0.9:
            insights.append("High confidence classification")
        elif confidence < 0.6:
            insights.append("Low confidence - may need more data")
        
        # Class-specific insights
        if self.classification_type == 'complexity':
            if predicted_class == 'simple':
                insights.append("Simple workflow - good candidate for automation")
            elif predicted_class == 'very_complex':
                insights.append("Very complex workflow - consider decomposition")
        elif self.classification_type == 'performance':
            if predicted_class == 'poor_performance':
                insights.append("Performance issues detected - optimization recommended")
            elif predicted_class == 'high_performance':
                insights.append("Well-optimized workflow")
        
        return "; ".join(insights) if insights else "Standard workflow"
        
    def _get_class_recommendations(
        self, 
        predicted_class: str, 
        confidence: float, 
        workflow_data: Optional[pd.Series] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations based on predicted class.
        
        Args:
            predicted_class: Predicted workflow class
            confidence: Classification confidence
            workflow_data: Original workflow data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if self.classification_type == 'complexity':
            if predicted_class == 'simple':
                recommendations.append({
                    'type': 'automation',
                    'description': 'Consider full automation for this simple workflow',
                    'priority': 'low',
                    'effort': 'low'
                })
            elif predicted_class == 'very_complex':
                recommendations.append({
                    'type': 'decomposition',
                    'description': 'Break down complex workflow into smaller components',
                    'priority': 'high',
                    'effort': 'high'
                })
                
        elif self.classification_type == 'performance':
            if predicted_class == 'poor_performance':
                recommendations.append({
                    'type': 'optimization',
                    'description': 'Implement performance optimizations',
                    'priority': 'high',
                    'effort': 'medium'
                })
            elif predicted_class == 'high_performance':
                recommendations.append({
                    'type': 'monitoring',
                    'description': 'Maintain current performance with enhanced monitoring',
                    'priority': 'low',
                    'effort': 'low'
                })
                
        elif self.classification_type == 'pattern':
            if predicted_class == 'unreliable':
                recommendations.append({
                    'type': 'reliability',
                    'description': 'Improve error handling and retry mechanisms',
                    'priority': 'high',
                    'effort': 'medium'
                })
        
        # Add confidence-based recommendations
        if confidence < 0.6:
            recommendations.append({
                'type': 'data_collection',
                'description': 'Collect more execution data for better classification',
                'priority': 'medium',
                'effort': 'low'
            })
        
        return recommendations
        
    def _evaluate_classifier(self, X: pd.DataFrame, y: np.ndarray):
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
        """
        try:
            # Split data for evaluation
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Refit on training data and evaluate on test data
                temp_model = RandomForestClassifier(
                    n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
                )
                temp_model.fit(X_train, y_train)
                y_pred = temp_model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                metrics = {
                    'accuracy': float(accuracy),
                    'macro_avg_f1': float(report['macro avg']['f1-score']),
                    'weighted_avg_f1': float(report['weighted avg']['f1-score']),
                    'n_classes': len(self.class_names)
                }
            else:
                # For small datasets, use training accuracy
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(y, y_pred)
                
                metrics = {
                    'accuracy': float(accuracy),
                    'n_classes': len(self.class_names),
                    'note': 'Training accuracy (small dataset)'
                }
            
            self.model_metadata['performance_metrics'] = metrics
            logger.info(f"Classification accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.warning(f"Error evaluating classifier: {e}")
