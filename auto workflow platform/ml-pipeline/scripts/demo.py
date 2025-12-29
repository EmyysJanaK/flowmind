"""
Demo Script for FlowMind ML Pipeline

Demonstrates the capabilities of the ML pipeline with real-world scenarios.
"""

import pandas as pd
import numpy as np
import sys
import os
import tempfile
import sqlite3
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import DataLoader, Preprocessor, FeatureEngineer
from models import ProcessPredictor, BottleneckDetector, OptimizationEngine, WorkflowClassifier
from utils import ModelMetrics, Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPipelineDemo:
    """Demo class for the ML pipeline."""
    
    def __init__(self):
        self.demo_db_path = None
        self.output_dir = None
        self.models = {}
        self.features = {}
        self.dataset = None
    
    def setup_demo_environment(self):
        """Setup demo environment with realistic data."""
        logger.info("Setting up demo environment...")
        
        # Create output directory
        self.output_dir = Path(__file__).parent.parent / 'demo_output'
        self.output_dir.mkdir(exist_ok=True)
        
        # Create demo database
        self.demo_db_path = self.output_dir / "demo_db.sqlite"
        self._create_realistic_database()
        
        logger.info(f"Demo environment created at: {self.output_dir}")
    
    def _create_realistic_database(self):
        """Create a realistic database with workflow scenarios."""
        logger.info("Creating realistic workflow database...")
        
        conn = sqlite3.connect(self.demo_db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE workflows (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE workflow_executions (
                id INTEGER PRIMARY KEY,
                workflow_id INTEGER,
                status TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration REAL,
                success_rate REAL,
                FOREIGN KEY (workflow_id) REFERENCES workflows (id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE execution_steps (
                id INTEGER PRIMARY KEY,
                execution_id INTEGER,
                step_name TEXT,
                status TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration REAL,
                resources_used TEXT,
                FOREIGN KEY (execution_id) REFERENCES workflow_executions (id)
            )
        """)
        
        # Insert realistic workflows
        workflows = [
            (1, "E-commerce Data Pipeline", "Processes customer orders and inventory data", "2024-01-01", "2024-01-01"),
            (2, "Financial Risk Assessment", "Analyzes trading risks and compliance", "2024-01-01", "2024-01-01"),
            (3, "Customer Analytics Pipeline", "Generates customer insights and recommendations", "2024-01-01", "2024-01-01"),
            (4, "Supply Chain Optimization", "Optimizes inventory and logistics", "2024-01-01", "2024-01-01"),
            (5, "Fraud Detection System", "Detects fraudulent transactions", "2024-01-01", "2024-01-01"),
            (6, "Marketing Campaign Analysis", "Analyzes campaign performance", "2024-01-01", "2024-01-01"),
            (7, "Content Recommendation Engine", "Generates personalized content", "2024-01-01", "2024-01-01"),
            (8, "Quality Control Monitoring", "Monitors product quality metrics", "2024-01-01", "2024-01-01")
        ]
        
        cursor.executemany(
            "INSERT INTO workflows (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            workflows
        )
        
        # Generate realistic execution patterns
        base_time = datetime.now() - timedelta(days=60)
        executions = []
        steps = []
        execution_id = 1
        step_id = 1
        
        # Define workflow characteristics
        workflow_patterns = {
            1: {'avg_duration': 45, 'success_rate': 0.92, 'steps': ['data_extraction', 'validation', 'transformation', 'loading']},
            2: {'avg_duration': 120, 'success_rate': 0.98, 'steps': ['data_collection', 'risk_calculation', 'model_scoring', 'report_generation']},
            3: {'avg_duration': 75, 'success_rate': 0.89, 'steps': ['customer_segmentation', 'behavior_analysis', 'recommendation_generation']},
            4: {'avg_duration': 90, 'success_rate': 0.94, 'steps': ['inventory_check', 'demand_forecasting', 'optimization', 'routing']},
            5: {'avg_duration': 30, 'success_rate': 0.96, 'steps': ['transaction_analysis', 'pattern_detection', 'scoring']},
            6: {'avg_duration': 60, 'success_rate': 0.91, 'steps': ['data_collection', 'performance_analysis', 'attribution_modeling']},
            7: {'avg_duration': 40, 'success_rate': 0.93, 'steps': ['user_profiling', 'content_scoring', 'recommendation_ranking']},
            8: {'avg_duration': 25, 'success_rate': 0.97, 'steps': ['sensor_data_collection', 'quality_scoring', 'alert_generation']}
        }
        
        # Generate executions for the last 60 days
        for day in range(60):
            current_date = base_time + timedelta(days=day)
            
            # Different workflows run at different frequencies
            daily_schedules = {
                1: 6,  # E-commerce: 6 times per day
                2: 2,  # Risk assessment: 2 times per day
                3: 3,  # Customer analytics: 3 times per day
                4: 4,  # Supply chain: 4 times per day
                5: 24, # Fraud detection: hourly
                6: 1,  # Marketing: once per day
                7: 12, # Recommendations: twice hourly
                8: 8   # Quality control: every 3 hours
            }
            
            for workflow_id, pattern in workflow_patterns.items():
                frequency = daily_schedules[workflow_id]
                
                for run in range(frequency):
                    # Calculate execution time
                    hour_offset = (24 / frequency) * run
                    start_time = current_date + timedelta(hours=hour_offset)
                    
                    # Add some randomness to execution times
                    start_time += timedelta(minutes=np.random.normal(0, 15))
                    
                    # Determine success/failure
                    is_successful = np.random.random() < pattern['success_rate']
                    
                    # Calculate duration with realistic variations
                    base_duration = pattern['avg_duration']
                    if is_successful:
                        duration = np.random.normal(base_duration, base_duration * 0.2)
                        status = 'completed'
                        success_rate = np.random.uniform(0.85, 1.0)
                    else:
                        duration = np.random.normal(base_duration * 0.6, base_duration * 0.1)
                        status = np.random.choice(['failed', 'cancelled'], p=[0.8, 0.2])
                        success_rate = np.random.uniform(0.0, 0.4)
                    
                    duration = max(5, duration)  # Minimum 5 minutes
                    end_time = start_time + timedelta(minutes=duration)
                    
                    executions.append((
                        execution_id, workflow_id, status,
                        start_time.isoformat(),
                        end_time.isoformat(),
                        duration, success_rate
                    ))
                    
                    # Generate steps for this execution
                    workflow_steps = pattern['steps']
                    step_duration = duration / len(workflow_steps)
                    
                    for i, step_name in enumerate(workflow_steps):
                        step_start = start_time + timedelta(minutes=i * step_duration)
                        
                        # Add realistic step variations
                        step_actual_duration = step_duration * np.random.uniform(0.7, 1.5)
                        step_end = step_start + timedelta(minutes=step_actual_duration)
                        
                        # Step may fail even if execution succeeds
                        step_success_rate = 0.98 if is_successful else 0.2
                        step_status = 'completed' if np.random.random() < step_success_rate else 'failed'
                        
                        # Generate realistic resource usage
                        cpu_usage = np.random.uniform(0.1, 0.8)
                        memory_usage = np.random.uniform(0.2, 0.7)
                        disk_io = np.random.uniform(0.1, 0.6)
                        
                        steps.append((
                            step_id, execution_id, step_name, step_status,
                            step_start.isoformat(), step_end.isoformat(),
                            step_actual_duration,
                            json.dumps({
                                "cpu": cpu_usage,
                                "memory": memory_usage,
                                "disk_io": disk_io,
                                "network": np.random.uniform(0.05, 0.3)
                            })
                        ))
                        
                        step_id += 1
                    
                    execution_id += 1
        
        # Insert data
        cursor.executemany(
            "INSERT INTO workflow_executions (id, workflow_id, status, start_time, end_time, duration, success_rate) VALUES (?, ?, ?, ?, ?, ?, ?)",
            executions
        )
        
        cursor.executemany(
            "INSERT INTO execution_steps (id, execution_id, step_name, status, start_time, end_time, duration, resources_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            steps
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created realistic database with {len(executions)} executions and {len(steps)} steps")
    
    def demonstrate_data_loading(self):
        """Demonstrate data loading capabilities."""
        print("\n" + "="*60)
        print("🔄 DEMONSTRATING DATA LOADING")
        print("="*60)
        
        database_url = f"sqlite:///{self.demo_db_path}"
        data_loader = DataLoader(database_url)
        
        # Load raw data
        print("Loading raw workflow data...")
        raw_data = data_loader.load_raw_data(lookback_days=30)
        
        print(f"📊 Loaded {len(raw_data['executions'])} executions")
        print(f"📊 Loaded {len(raw_data.get('steps', []))} execution steps")
        
        # Show execution status distribution
        status_counts = raw_data['executions']['status'].value_counts()
        print("\n📈 Execution Status Distribution:")
        for status, count in status_counts.items():
            percentage = (count / len(raw_data['executions'])) * 100
            print(f"   {status}: {count} ({percentage:.1f}%)")
        
        # Create training dataset
        print("\nCreating training dataset...")
        self.dataset = data_loader.create_training_dataset(lookback_days=30, min_executions=5)
        
        print(f"📊 Training dataset created with {self.dataset['metadata']['num_executions']} executions")
        print(f"📊 Covering {self.dataset['metadata']['num_workflows']} workflows")
        
        return self.dataset
    
    def demonstrate_preprocessing(self):
        """Demonstrate data preprocessing."""
        print("\n" + "="*60)
        print("🔧 DEMONSTRATING DATA PREPROCESSING")
        print("="*60)
        
        preprocessor = Preprocessor()
        
        # Prepare data for training
        print("Preprocessing execution data...")
        processed_data = preprocessor.prepare_for_training(
            self.dataset['executions'],
            self.dataset.get('steps')
        )
        
        print(f"📊 Processed {len(processed_data['executions'])} executions")
        
        # Show data quality improvements
        original_nulls = self.dataset['executions'].isnull().sum().sum()
        processed_nulls = processed_data['executions'].isnull().sum().sum()
        
        print(f"📈 Data Quality Improvement:")
        print(f"   Original null values: {original_nulls}")
        print(f"   After preprocessing: {processed_nulls}")
        print(f"   Improvement: {((original_nulls - processed_nulls) / max(original_nulls, 1)) * 100:.1f}%")
        
        # Demonstrate scaling
        numeric_cols = processed_data['executions'].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n🔧 Scaling {len(numeric_cols)} numeric features...")
            scaled_data = preprocessor.scale_features(processed_data['executions'][numeric_cols])
            
            print("📈 Feature scaling applied:")
            print(f"   Mean after scaling: {scaled_data.mean().mean():.4f}")
            print(f"   Std after scaling: {scaled_data.std().mean():.4f}")
        
        return processed_data
    
    def demonstrate_feature_engineering(self):
        """Demonstrate feature engineering capabilities."""
        print("\n" + "="*60)
        print("🛠️  DEMONSTRATING FEATURE ENGINEERING")
        print("="*60)
        
        feature_engineer = FeatureEngineer()
        
        # Create different types of features
        print("Creating workflow-level features...")
        workflow_features = feature_engineer.create_workflow_features(self.dataset['executions'])
        print(f"📊 Generated {len(workflow_features.columns)} workflow features")
        
        print("\nCreating execution-level features...")
        execution_features = feature_engineer.create_execution_features(
            self.dataset['executions'],
            self.dataset.get('steps')
        )
        print(f"📊 Generated {len(execution_features.columns)} execution features")
        
        print("\nCreating prediction features...")
        prediction_features = feature_engineer.create_prediction_features(self.dataset['executions'])
        print(f"📊 Generated {len(prediction_features.columns)} prediction features")
        
        print("\nCreating clustering features...")
        clustering_features = feature_engineer.create_clustering_features(self.dataset['executions'])
        print(f"📊 Generated {len(clustering_features.columns)} clustering features")
        
        # Bottleneck features
        bottleneck_features = None
        if 'steps' in self.dataset and len(self.dataset['steps']) > 0:
            print("\nCreating bottleneck features...")
            bottleneck_features = feature_engineer.create_bottleneck_features(self.dataset['steps'])
            print(f"📊 Generated {len(bottleneck_features.columns)} bottleneck features")
        
        # Show feature examples
        print("\n📈 Sample Features Created:")
        feature_examples = []
        if len(workflow_features.columns) > 0:
            feature_examples.extend(list(workflow_features.columns)[:3])
        if len(prediction_features.columns) > 0:
            feature_examples.extend(list(prediction_features.columns)[:3])
        
        for feature in feature_examples[:5]:
            print(f"   • {feature}")
        
        self.features = {
            'workflow_features': workflow_features,
            'execution_features': execution_features,
            'prediction_features': prediction_features,
            'clustering_features': clustering_features,
            'bottleneck_features': bottleneck_features
        }
        
        return self.features
    
    def demonstrate_model_training(self):
        """Demonstrate model training capabilities."""
        print("\n" + "="*60)
        print("🤖 DEMONSTRATING MODEL TRAINING")
        print("="*60)
        
        # Prepare training data
        X = self.features['prediction_features'].select_dtypes(include=[np.number])
        y = pd.DataFrame({
            'duration': self.dataset['executions']['duration'],
            'success_rate': (self.dataset['executions']['status'] == 'completed').astype(float)
        })
        
        print(f"📊 Training data: {len(X)} samples, {len(X.columns)} features")
        
        # Train Process Predictor
        print("\n🔮 Training Process Predictor...")
        process_predictor = ProcessPredictor('random_forest')
        process_predictor.fit(X, y)
        
        # Get feature importance
        importance = process_predictor.get_feature_importance()
        if 'duration' in importance:
            top_features = sorted(importance['duration'].items(), key=lambda x: x[1], reverse=True)[:5]
            print("📈 Top 5 Features for Duration Prediction:")
            for feature, score in top_features:
                print(f"   • {feature}: {score:.4f}")
        
        self.models['process_predictor'] = process_predictor
        
        # Train Bottleneck Detector
        if self.features['bottleneck_features'] is not None:
            print("\n🔍 Training Bottleneck Detector...")
            X_bottleneck = self.features['bottleneck_features'].select_dtypes(include=[np.number])
            
            bottleneck_detector = BottleneckDetector('isolation_forest')
            bottleneck_detector.fit(X_bottleneck)
            
            # Analyze bottlenecks
            analysis = bottleneck_detector.analyze_bottlenecks(X_bottleneck, self.dataset.get('steps', pd.DataFrame()))
            if analysis and 'bottleneck_steps' in analysis:
                print(f"📊 Identified {len(analysis['bottleneck_steps'])} potential bottleneck steps")
            
            self.models['bottleneck_detector'] = bottleneck_detector
        
        # Train Optimization Engine
        print("\n⚡ Training Optimization Engine...")
        optimization_engine = OptimizationEngine('performance')
        optimization_engine.fit(self.features['workflow_features'], y)
        
        # Generate sample recommendations
        recommendations = optimization_engine.generate_recommendations(
            self.features['workflow_features'],
            self.dataset['executions'],
            self.features['bottleneck_features']
        )
        
        if recommendations:
            print(f"📊 Generated {len(recommendations)} optimization recommendations")
            print("📈 Sample Recommendation Types:")
            rec_types = set(rec.get('type', 'unknown') for rec in recommendations)
            for rec_type in list(rec_types)[:3]:
                print(f"   • {rec_type}")
        
        self.models['optimization_engine'] = optimization_engine
        
        # Train Workflow Classifier
        print("\n🏷️  Training Workflow Classifier...")
        workflow_classifier = WorkflowClassifier('complexity')
        workflow_classifier.fit(self.features['workflow_features'])
        
        # Perform classification
        classifications = workflow_classifier.classify_workflows(self.features['workflow_features'])
        complexity_dist = classifications['complexity_category'].value_counts()
        
        print("📈 Workflow Complexity Distribution:")
        for category, count in complexity_dist.items():
            print(f"   • {category}: {count} workflows")
        
        self.models['workflow_classifier'] = workflow_classifier
        
        print(f"\n✅ Successfully trained {len(self.models)} models!")
    
    def demonstrate_predictions(self):
        """Demonstrate making predictions with trained models."""
        print("\n" + "="*60)
        print("🔮 DEMONSTRATING PREDICTIONS & INSIGHTS")
        print("="*60)
        
        # Process Predictions
        if 'process_predictor' in self.models:
            print("🔮 Process Duration & Success Predictions...")
            
            X = self.features['prediction_features'].select_dtypes(include=[np.number])
            predictions = self.models['process_predictor'].predict(X)
            
            if 'duration' in predictions.columns:
                avg_predicted_duration = predictions['duration'].mean()
                actual_avg_duration = self.dataset['executions']['duration'].mean()
                
                print(f"📊 Average Predicted Duration: {avg_predicted_duration:.1f} minutes")
                print(f"📊 Actual Average Duration: {actual_avg_duration:.1f} minutes")
                print(f"📊 Prediction Accuracy: {(1 - abs(avg_predicted_duration - actual_avg_duration) / actual_avg_duration) * 100:.1f}%")
            
            if 'success_rate' in predictions.columns:
                avg_predicted_success = predictions['success_rate'].mean()
                actual_success_rate = (self.dataset['executions']['status'] == 'completed').mean()
                
                print(f"📊 Average Predicted Success Rate: {avg_predicted_success:.3f}")
                print(f"📊 Actual Success Rate: {actual_success_rate:.3f}")
        
        # Bottleneck Detection
        if 'bottleneck_detector' in self.models and self.features['bottleneck_features'] is not None:
            print("\n🔍 Bottleneck Detection Results...")
            
            X_bottleneck = self.features['bottleneck_features'].select_dtypes(include=[np.number])
            bottleneck_predictions = self.models['bottleneck_detector'].predict(X_bottleneck)
            
            # Count anomalies
            if hasattr(bottleneck_predictions, '__len__'):
                anomaly_count = sum(1 for pred in bottleneck_predictions if pred == -1)
                anomaly_rate = (anomaly_count / len(bottleneck_predictions)) * 100
                
                print(f"📊 Detected {anomaly_count} potential bottlenecks ({anomaly_rate:.1f}% of steps)")
        
        # Optimization Recommendations
        if 'optimization_engine' in self.models:
            print("\n⚡ Optimization Recommendations...")
            
            recommendations = self.models['optimization_engine'].generate_recommendations(
                self.features['workflow_features'],
                self.dataset['executions'],
                self.features['bottleneck_features']
            )
            
            if recommendations:
                print(f"📊 Generated {len(recommendations)} recommendations")
                
                # Show sample recommendations
                print("📈 Sample Recommendations:")
                for i, rec in enumerate(recommendations[:3]):
                    impact = rec.get('potential_impact', {})
                    print(f"   {i+1}. {rec.get('description', 'Optimization recommendation')}")
                    if 'time_savings' in impact:
                        print(f"      💰 Potential time savings: {impact['time_savings']:.1f} minutes")
                    if 'success_rate_improvement' in impact:
                        print(f"      📈 Success rate improvement: +{impact['success_rate_improvement']:.1%}")
        
        # Workflow Classification
        if 'workflow_classifier' in self.models:
            print("\n🏷️  Workflow Classification Insights...")
            
            classifications = self.models['workflow_classifier'].classify_workflows(
                self.features['workflow_features']
            )
            
            # Show insights by complexity
            complexity_stats = classifications.groupby('complexity_category').agg({
                'avg_duration': 'mean',
                'success_rate': 'mean'
            }).round(2)
            
            print("📈 Performance by Workflow Complexity:")
            for category, stats in complexity_stats.iterrows():
                print(f"   • {category}:")
                print(f"     Average Duration: {stats['avg_duration']:.1f} minutes")
                print(f"     Success Rate: {stats['success_rate']:.3f}")
    
    def demonstrate_visualizations(self):
        """Demonstrate visualization capabilities."""
        print("\n" + "="*60)
        print("📊 DEMONSTRATING VISUALIZATIONS")
        print("="*60)
        
        visualizer = Visualizer()
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        print("Creating visualizations...")
        
        # 1. Workflow Performance Timeline
        if 'start_time' in self.dataset['executions'].columns:
            print("📈 Creating workflow performance timeline...")
            fig = visualizer.plot_workflow_performance_timeline(
                self.dataset['executions'],
                save_path=str(plots_dir / 'workflow_timeline.png')
            )
            plt.close(fig)
        
        # 2. Feature Importance
        if 'process_predictor' in self.models:
            importance = self.models['process_predictor'].get_feature_importance()
            if 'duration' in importance:
                print("📈 Creating feature importance plot...")
                fig = visualizer.plot_feature_importance(
                    importance['duration'],
                    save_path=str(plots_dir / 'feature_importance.png')
                )
                plt.close(fig)
        
        # 3. Bottleneck Analysis
        if self.features['bottleneck_features'] is not None:
            print("📈 Creating bottleneck analysis...")
            bottleneck_data = self.features['bottleneck_features'].copy()
            
            if 'bottleneck_detector' in self.models:
                predictions = self.models['bottleneck_detector'].predict(
                    bottleneck_data.select_dtypes(include=[np.number])
                )
                bottleneck_data['is_bottleneck'] = predictions
            
            fig = visualizer.plot_bottleneck_analysis(
                bottleneck_data,
                save_path=str(plots_dir / 'bottleneck_analysis.png')
            )
            plt.close(fig)
        
        # 4. Optimization Recommendations
        if 'optimization_engine' in self.models:
            recommendations = self.models['optimization_engine'].generate_recommendations(
                self.features['workflow_features'],
                self.dataset['executions'],
                self.features['bottleneck_features']
            )
            
            if recommendations:
                print("📈 Creating optimization recommendations plot...")
                fig = visualizer.plot_optimization_recommendations(
                    recommendations,
                    save_path=str(plots_dir / 'optimization_recommendations.png')
                )
                plt.close(fig)
        
        # 5. Comprehensive Dashboard
        print("📈 Creating comprehensive dashboard...")
        model_results = {}
        if 'process_predictor' in self.models:
            # Get some metrics for the dashboard
            X = self.features['prediction_features'].select_dtypes(include=[np.number])
            predictions = self.models['process_predictor'].predict(X)
            
            if 'duration' in predictions.columns:
                from utils import ModelMetrics
                metrics = ModelMetrics()
                y_true = self.dataset['executions']['duration']
                y_pred = predictions['duration']
                
                model_results['process_predictor'] = {
                    'r2': metrics.r2_score(y_true, y_pred),
                    'mae': metrics.mean_absolute_error(y_true, y_pred)
                }
        
        fig = visualizer.create_dashboard(
            model_results,
            self.dataset['executions'],
            self.features['bottleneck_features'],
            save_path=str(plots_dir / 'dashboard.png')
        )
        plt.close(fig)
        
        print(f"✅ All visualizations saved to: {plots_dir}")
        
        # List created files
        plot_files = list(plots_dir.glob('*.png'))
        print("\n📊 Created Visualizations:")
        for plot_file in plot_files:
            print(f"   • {plot_file.name}")
    
    def demonstrate_model_evaluation(self):
        """Demonstrate model evaluation capabilities."""
        print("\n" + "="*60)
        print("📏 DEMONSTRATING MODEL EVALUATION")
        print("="*60)
        
        from utils import ModelMetrics
        metrics = ModelMetrics()
        
        # Evaluate Process Predictor
        if 'process_predictor' in self.models:
            print("📏 Evaluating Process Predictor...")
            
            X = self.features['prediction_features'].select_dtypes(include=[np.number])
            y_duration = self.dataset['executions']['duration']
            y_success = (self.dataset['executions']['status'] == 'completed').astype(float)
            
            predictions = self.models['process_predictor'].predict(X)
            
            if 'duration' in predictions.columns:
                r2 = metrics.r2_score(y_duration, predictions['duration'])
                mae = metrics.mean_absolute_error(y_duration, predictions['duration'])
                mape = metrics.mean_absolute_percentage_error(y_duration, predictions['duration'])
                
                print(f"   📊 Duration Prediction R²: {r2:.4f}")
                print(f"   📊 Duration Prediction MAE: {mae:.2f} minutes")
                print(f"   📊 Duration Prediction MAPE: {mape:.2%}")
            
            if 'success_rate' in predictions.columns:
                accuracy = metrics.accuracy_score(y_success, predictions['success_rate'] > 0.5)
                auc = metrics.roc_auc_score(y_success, predictions['success_rate'])
                
                print(f"   📊 Success Prediction Accuracy: {accuracy:.4f}")
                print(f"   📊 Success Prediction AUC: {auc:.4f}")
            
            # Cross-validation
            print("   🔄 Performing cross-validation...")
            cv_scores = self.models['process_predictor'].cross_validate(
                X, pd.DataFrame({'duration': y_duration, 'success_rate': y_success})
            )
            
            if cv_scores:
                for metric, scores in cv_scores.items():
                    if isinstance(scores, list) and len(scores) > 0:
                        mean_score = np.mean(scores)
                        std_score = np.std(scores)
                        print(f"   📊 CV {metric}: {mean_score:.4f} (±{std_score:.4f})")
        
        # Save evaluation results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.models.keys()),
            'dataset_size': len(self.dataset['executions']),
            'feature_count': len(self.features['prediction_features'].columns)
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"\n✅ Evaluation results saved to: {self.output_dir / 'evaluation_results.json'}")
    
    def generate_demo_report(self):
        """Generate comprehensive demo report."""
        print("\n" + "="*60)
        print("📋 GENERATING DEMO REPORT")
        print("="*60)
        
        report = {
            'demo_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_executions': len(self.dataset['executions']),
                'total_workflows': self.dataset['metadata']['num_workflows'],
                'date_range_days': 30,
                'success_rate': (self.dataset['executions']['status'] == 'completed').mean()
            },
            'models_trained': list(self.models.keys()),
            'features_generated': {
                'workflow_features': len(self.features['workflow_features'].columns),
                'execution_features': len(self.features['execution_features'].columns),
                'prediction_features': len(self.features['prediction_features'].columns),
                'clustering_features': len(self.features['clustering_features'].columns),
                'bottleneck_features': len(self.features['bottleneck_features'].columns) if self.features['bottleneck_features'] is not None else 0
            },
            'capabilities_demonstrated': [
                'Data Loading from SQLite Database',
                'Data Preprocessing and Cleaning',
                'Feature Engineering (Multiple Types)',
                'Model Training (4 Different Models)',
                'Prediction Generation',
                'Model Evaluation and Metrics',
                'Visualization Generation',
                'Optimization Recommendations',
                'Bottleneck Detection',
                'Workflow Classification'
            ]
        }
        
        # Save report
        report_path = self.output_dir / 'demo_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_path = self.output_dir / 'demo_report.html'
        with open(html_path, 'w') as f:
            f.write(html_report)
        
        print(f"📋 Demo report saved to: {report_path}")
        print(f"📋 HTML report saved to: {html_path}")
        
        return report
    
    def _generate_html_report(self, report):
        """Generate HTML demo report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FlowMind ML Pipeline Demo Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .metric {{ background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .success {{ color: #27ae60; font-weight: bold; }}
                .highlight {{ background: #f1c40f; padding: 2px 6px; border-radius: 3px; }}
                ul {{ list-style-type: none; padding: 0; }}
                li {{ background: #e8f5e8; margin: 5px 0; padding: 10px; border-left: 4px solid #27ae60; }}
                .stats {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
                .stat-box {{ background: #3498db; color: white; padding: 20px; margin: 10px; border-radius: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>🚀 FlowMind ML Pipeline Demo Report</h1>
            <p><strong>Generated:</strong> {report['demo_timestamp']}</p>
            
            <h2>📊 Dataset Overview</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>{report['dataset_summary']['total_executions']}</h3>
                    <p>Total Executions</p>
                </div>
                <div class="stat-box">
                    <h3>{report['dataset_summary']['total_workflows']}</h3>
                    <p>Workflows Analyzed</p>
                </div>
                <div class="stat-box">
                    <h3>{report['dataset_summary']['success_rate']:.1%}</h3>
                    <p>Success Rate</p>
                </div>
                <div class="stat-box">
                    <h3>{report['dataset_summary']['date_range_days']}</h3>
                    <p>Days of Data</p>
                </div>
            </div>
            
            <h2>🤖 Models Successfully Trained</h2>
            <ul>
        """
        
        model_descriptions = {
            'process_predictor': 'Predicts workflow execution duration and success probability',
            'bottleneck_detector': 'Identifies performance bottlenecks in workflow steps',
            'optimization_engine': 'Generates actionable optimization recommendations',
            'workflow_classifier': 'Classifies workflows by complexity and patterns'
        }
        
        for model in report['models_trained']:
            description = model_descriptions.get(model, 'Advanced ML model')
            html += f'<li><strong>{model.replace("_", " ").title()}</strong>: {description}</li>'
        
        html += f"""
            </ul>
            
            <h2>🛠️ Features Generated</h2>
            <div class="metric">
                <strong>Workflow Features:</strong> {report['features_generated']['workflow_features']} features for workflow-level analysis
            </div>
            <div class="metric">
                <strong>Execution Features:</strong> {report['features_generated']['execution_features']} features for execution-level insights
            </div>
            <div class="metric">
                <strong>Prediction Features:</strong> {report['features_generated']['prediction_features']} features optimized for predictive modeling
            </div>
            <div class="metric">
                <strong>Clustering Features:</strong> {report['features_generated']['clustering_features']} features for pattern discovery
            </div>
            <div class="metric">
                <strong>Bottleneck Features:</strong> {report['features_generated']['bottleneck_features']} features for bottleneck detection
            </div>
            
            <h2>✨ Capabilities Demonstrated</h2>
            <ul>
        """
        
        for capability in report['capabilities_demonstrated']:
            html += f'<li>{capability}</li>'
        
        html += """
            </ul>
            
            <h2>🎯 Key Achievements</h2>
            <div class="metric">
                <span class="success">✅ Complete End-to-End Pipeline:</span> From raw data to actionable insights
            </div>
            <div class="metric">
                <span class="success">✅ Multi-Model Architecture:</span> Specialized models for different use cases
            </div>
            <div class="metric">
                <span class="success">✅ Comprehensive Feature Engineering:</span> Advanced feature creation for optimal performance
            </div>
            <div class="metric">
                <span class="success">✅ Production-Ready Code:</span> Robust error handling and logging
            </div>
            <div class="metric">
                <span class="success">✅ Rich Visualizations:</span> Interactive plots and dashboards
            </div>
            
            <h2>📈 Next Steps</h2>
            <p>The ML pipeline is now ready for:</p>
            <ul>
                <li><strong>Production Deployment:</strong> Integrate with your existing workflow management system</li>
                <li><strong>Real-time Predictions:</strong> Deploy models for live workflow optimization</li>
                <li><strong>Continuous Learning:</strong> Set up automated retraining with new data</li>
                <li><strong>Advanced Analytics:</strong> Extend with additional models and features</li>
                <li><strong>Integration Testing:</strong> Connect with your actual database and workflows</li>
            </ul>
            
        </body>
        </html>
        """
        
        return html
    
    def run_complete_demo(self):
        """Run the complete demo."""
        print("🚀 Starting FlowMind ML Pipeline Demo")
        print("="*60)
        
        try:
            # Setup
            self.setup_demo_environment()
            
            # Demonstrate each component
            self.demonstrate_data_loading()
            self.demonstrate_preprocessing()
            self.demonstrate_feature_engineering()
            self.demonstrate_model_training()
            self.demonstrate_predictions()
            self.demonstrate_model_evaluation()
            self.demonstrate_visualizations()
            
            # Generate final report
            report = self.generate_demo_report()
            
            # Final summary
            print("\n" + "="*60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📊 Processed {report['dataset_summary']['total_executions']} workflow executions")
            print(f"🤖 Trained {len(report['models_trained'])} ML models")
            print(f"🛠️ Generated {sum(report['features_generated'].values())} features")
            print(f"📈 Created comprehensive visualizations and insights")
            print(f"📁 All results saved to: {self.output_dir}")
            
            print("\n🎯 The FlowMind ML Pipeline is ready for production!")
            print("   • Models are trained and validated")
            print("   • Visualizations provide actionable insights")
            print("   • Complete documentation and reports generated")
            print("   • Ready for integration with your workflow system")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Demo failed")
            raise


def main():
    """Main demo function."""
    demo = MLPipelineDemo()
    
    try:
        demo.run_complete_demo()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
