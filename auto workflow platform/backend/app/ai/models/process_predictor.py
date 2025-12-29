import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class ProcessPredictor:
    def __init__(self):
        self.performance_model = None
        self.bottleneck_model = None
        self.optimization_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    async def train_models(self, training_data: pd.DataFrame):
        """Train all prediction models"""
        
        # Prepare features
        features = self._extract_features(training_data)
        
        # Train performance prediction model
        performance_targets = training_data['performance_score']
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.performance_model.fit(features, performance_targets)
        
        # Train bottleneck detection model
        bottleneck_targets = training_data['has_bottleneck']
        self.bottleneck_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.bottleneck_model.fit(features, bottleneck_targets)
        
        # Train optimization model
        optimization_targets = training_data['optimization_potential']
        self.optimization_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.optimization_model.fit(features, optimization_targets)
        
        self.is_trained = True
        
        # Save models
        self._save_models()
    
    async def predict_performance_score(self, workflow_definition: Dict[str, Any]) -> float:
        """Predict workflow performance score"""
        
        if not self.is_trained:
            await self._load_or_initialize_models()
        
        features = self._extract_workflow_features(workflow_definition)
        
        if self.performance_model:
            score = self.performance_model.predict([features])[0]
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        
        return 0.5  # Default score
    
    async def predict_bottlenecks(self, execution_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict potential bottlenecks in workflow execution"""
        
        if not self.is_trained:
            await self._load_or_initialize_models()
        
        features = self._extract_execution_features(execution_data)
        
        bottlenecks = []
        if self.bottleneck_model:
            bottleneck_probability = self.bottleneck_model.predict_proba([features])[0][1]
            
            if bottleneck_probability > 0.7:
                bottlenecks.append({
                    "type": "predicted_bottleneck",
                    "probability": bottleneck_probability,
                    "location": self._identify_bottleneck_location(features),
                    "severity": "high" if bottleneck_probability > 0.9 else "medium"
                })
        
        return bottlenecks
    
    async def predict_optimizations(self, execution) -> List[Dict[str, Any]]:
        """Predict optimization opportunities"""
        
        if not self.is_trained:
            await self._load_or_initialize_models()
        
        # Extract features from execution
        features = self._extract_execution_features_from_model(execution)
        
        optimizations = []
        if self.optimization_model:

            optimization_potential = self.optimization_model.predict([features])[0]
            
            if optimization_potential > 0.6:

                optimizations.extend([
                    {
                        "type": "parallelization",
                        "description": "Parallelize independent steps to reduce execution time",
                        "confidence": optimization_potential * 0.8,
                        "expected_improvement": f"{optimization_potential * 30:.1f}%"
                    },
                    {
                        "type": "caching",
                        "description": "Cache frequently accessed data to improve performance",
                        "confidence": optimization_potential * 0.6,
                        "expected_improvement": f"{optimization_potential * 20:.1f}%"
                    },
                    {
                        "type": "resource_optimization",
                        "description": "Optimize resource allocation based on workload patterns",
                        "confidence": optimization_potential * 0.7,
                        "expected_improvement": f"{optimization_potential * 25:.1f}%"
                    }
                ])
        
        return optimizations
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from training data"""
        features = []
        
        # Workflow complexity features
        features.append(data['num_steps'].values)
        features.append(data['num_dependencies'].values)
        features.append(data['avg_step_duration'].values)
        features.append(data['total_duration'].values)
        features.append(data['num_api_calls'].values)
        features.append(data['num_decision_points'].values)
        
        return np.column_stack(features)
    
    def _extract_workflow_features(self, workflow_definition: Dict[str, Any]) -> List[float]:
        """Extract features from workflow definition"""
        steps = workflow_definition.get("steps", [])
        
        features = [
            len(steps),  # Number of steps
            self._count_dependencies(steps),  # Number of dependencies
            self._calculate_complexity_score(steps),  # Complexity score
            self._count_api_calls(steps),  # Number of API calls
            self._count_decision_points(steps),  # Number of decision points
            self._calculate_parallel_potential(steps),  # Parallelization potential
        ]
        
        return features
    
    def _extract_execution_features(self, execution_data: Dict[str, Any]) -> List[float]:
        """Extract features from execution data"""
        return [
            execution_data.get('duration', 0),
            execution_data.get('num_steps', 0),
            execution_data.get('num_failures', 0),
            execution_data.get('avg_step_duration', 0),
            execution_data.get('resource_usage', 0),
            execution_data.get('concurrent_executions', 0),
        ]
    
    def _extract_execution_features_from_model(self, execution) -> List[float]:
        """Extract features from execution model"""
        features = [
            execution.duration or 0,
            len(execution.steps) if execution.steps else 0,
            len([s for s in execution.steps if s.status == 'failed']) if execution.steps else 0,
            np.mean([s.duration for s in execution.steps if s.duration]) if execution.steps else 0,
            1.0,  # Resource usage placeholder
            1.0,  # Concurrent executions placeholder
        ]
        
        return features
    
    def _count_dependencies(self, steps: List[Dict[str, Any]]) -> int:
        """Count dependencies between steps"""
        dependencies = 0
        for step in steps:
            if "depends_on" in step:
                dependencies += len(step["depends_on"])
        return dependencies
    
    def _calculate_complexity_score(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate workflow complexity score"""
        complexity = 0
        for step in steps:
            if step.get("type") == "decision":
                complexity += 2
            elif step.get("type") == "loop":
                complexity += 3
            else:
                complexity += 1
        return complexity / len(steps) if steps else 0
    
    def _count_api_calls(self, steps: List[Dict[str, Any]]) -> int:
        """Count API calls in workflow"""
        return len([s for s in steps if s.get("type") == "api_call"])
    
    def _count_decision_points(self, steps: List[Dict[str, Any]]) -> int:
        """Count decision points in workflow"""
        return len([s for s in steps if s.get("type") == "decision"])
    
    def _calculate_parallel_potential(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate parallelization potential"""
        independent_steps = 0
        for step in steps:
            if not step.get("depends_on"):
                independent_steps += 1
        return independent_steps / len(steps) if steps else 0
    
    def _identify_bottleneck_location(self, features: List[float]) -> str:
        """Identify likely bottleneck location"""
        if features[3] > 10:  # High number of API calls
            return "external_api"
        elif features[2] > 0.5:  # High complexity
            return "complex_logic"
        else:
            return "resource_constraint"
    
    def _save_models(self):
        """Save trained models"""
        if self.performance_model:
            joblib.dump(self.performance_model, 'models/performance_model.pkl')
        if self.bottleneck_model:
            joblib.dump(self.bottleneck_model, 'models/bottleneck_model.pkl')
        if self.optimization_model:
            joblib.dump(self.optimization_model, 'models/optimization_model.pkl')
    
    async def _load_or_initialize_models(self):
        """Load existing models or initialize with defaults"""
        try:
            self.performance_model = joblib.load('models/performance_model.pkl')
            self.bottleneck_model = joblib.load('models/bottleneck_model.pkl')
            self.optimization_model = joblib.load('models/optimization_model.pkl')
            self.is_trained = True
        except FileNotFoundError:
            # Initialize with dummy data for demo purposes
            await self._initialize_demo_models()
    
    async def _initialize_demo_models(self):
        """Initialize models with synthetic data for demonstration"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic features
        num_steps = np.random.randint(3, 20, n_samples)
        num_dependencies = np.random.randint(0, num_steps)
        complexity = np.random.uniform(0.2, 2.0, n_samples)
        num_api_calls = np.random.randint(0, 10, n_samples)
        
        features = np.column_stack([num_steps, num_dependencies, complexity, num_api_calls])
        
        # Generate synthetic targets
        performance_scores = 1.0 - (complexity * 0.3 + num_api_calls * 0.1) / 10
        performance_scores = np.clip(performance_scores, 0, 1)
        
        bottleneck_targets = (complexity > 1.5) | (num_api_calls > 5)
        optimization_potential = np.random.uniform(0.3, 0.9, n_samples)
        
        # Train models
        self.performance_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.performance_model.fit(features, performance_scores)
        
        self.bottleneck_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        self.bottleneck_model.fit(features, bottleneck_targets)
        
        self.optimization_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.optimization_model.fit(features, optimization_potential)
        
        self.is_trained = True