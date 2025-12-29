"""
Comprehensive Testing Script for FlowMind ML Pipeline

Tests the ML pipeline step by step to ensure everything works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path
import logging
import sqlite3
from datetime import datetime, timedelta
import json

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


class MLPipelineTester:
    """Comprehensive tester for the ML pipeline."""
    
    def __init__(self):
        self.test_db_path = None
        self.temp_dir = None
        self.results = {
            'data_loading': {},
            'preprocessing': {},
            'feature_engineering': {},
            'model_training': {},
            'model_evaluation': {},
            'utilities': {},
            'integration': {}
        }
    
    def setup_test_environment(self):
        """Setup test environment with sample data."""
        logger.info("Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Test directory: {self.temp_dir}")
        
        # Create test database
        self.test_db_path = self.temp_dir / "test_db.sqlite"
        self._create_test_database()
        
        logger.info("Test environment setup complete")
    
    def _create_test_database(self):
        """Create a test SQLite database with sample workflow data."""
        logger.info("Creating test database with sample data...")
        
        conn = sqlite3.connect(self.test_db_path)
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
        
        # Insert sample workflows
        workflows_data = [
            (1, "Data Processing Workflow", "Processes customer data", "2024-01-01 00:00:00", "2024-01-01 00:00:00"),
            (2, "ML Training Pipeline", "Trains machine learning models", "2024-01-01 00:00:00", "2024-01-01 00:00:00"),
            (3, "Report Generation", "Generates daily reports", "2024-01-01 00:00:00", "2024-01-01 00:00:00"),
            (4, "Data Backup Process", "Backs up database", "2024-01-01 00:00:00", "2024-01-01 00:00:00"),
            (5, "ETL Pipeline", "Extract, Transform, Load pipeline", "2024-01-01 00:00:00", "2024-01-01 00:00:00")
        ]
        
        cursor.executemany(
            "INSERT INTO workflows (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            workflows_data
        )
        
        # Insert sample executions
        base_time = datetime.now() - timedelta(days=30)
        executions_data = []
        steps_data = []
        
        for i in range(100):  # 100 executions
            workflow_id = (i % 5) + 1
            status = np.random.choice(['completed', 'failed', 'running'], p=[0.7, 0.2, 0.1])
            
            start_time = base_time + timedelta(hours=i*2.4)  # Spread over 30 days
            
            if status == 'completed':
                duration = np.random.normal(120, 30)  # 2 hours average
                success_rate = np.random.uniform(0.8, 1.0)
            elif status == 'failed':
                duration = np.random.normal(60, 20)  # Failed earlier
                success_rate = np.random.uniform(0.0, 0.3)
            else:  # running
                duration = None
                success_rate = None
            
            end_time = start_time + timedelta(minutes=duration) if duration else None
            
            executions_data.append((
                i + 1, workflow_id, status,
                start_time.isoformat(),
                end_time.isoformat() if end_time else None,
                duration, success_rate
            ))
            
            # Add steps for completed executions
            if status == 'completed' and duration:
                num_steps = np.random.randint(3, 8)
                step_duration = duration / num_steps
                
                for j in range(num_steps):
                    step_start = start_time + timedelta(minutes=j*step_duration)
                    step_end = step_start + timedelta(minutes=step_duration * np.random.uniform(0.8, 1.2))
                    step_status = 'completed' if np.random.random() > 0.1 else 'failed'
                    
                    steps_data.append((
                        len(steps_data) + 1,
                        i + 1,
                        f"Step_{j+1}",
                        step_status,
                        step_start.isoformat(),
                        step_end.isoformat(),
                        (step_end - step_start).total_seconds() / 60,
                        json.dumps({"cpu": np.random.uniform(0.1, 0.9), "memory": np.random.uniform(0.2, 0.8)})
                    ))
        
        cursor.executemany(
            "INSERT INTO workflow_executions (id, workflow_id, status, start_time, end_time, duration, success_rate) VALUES (?, ?, ?, ?, ?, ?, ?)",
            executions_data
        )
        
        cursor.executemany(
            "INSERT INTO execution_steps (id, execution_id, step_name, status, start_time, end_time, duration, resources_used) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            steps_data
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created test database with {len(executions_data)} executions and {len(steps_data)} steps")
    
    def test_data_loading(self):
        """Test data loading functionality."""
        logger.info("Testing data loading...")
        
        try:
            # Test DataLoader initialization
            database_url = f"sqlite:///{self.test_db_path}"
            data_loader = DataLoader(database_url)
            
            # Test loading raw data
            raw_data = data_loader.load_raw_data(lookback_days=30)
            assert 'executions' in raw_data, "Executions data not loaded"
            assert len(raw_data['executions']) > 0, "No execution data loaded"
            
            # Test creating training dataset
            dataset = data_loader.create_training_dataset(lookback_days=30, min_executions=1)
            assert dataset is not None, "Training dataset not created"
            assert 'metadata' in dataset, "Dataset metadata missing"
            
            self.results['data_loading'] = {
                'status': 'PASSED',
                'executions_loaded': len(raw_data['executions']),
                'steps_loaded': len(raw_data.get('steps', [])),
                'dataset_created': True
            }
            
            logger.info("✓ Data loading tests passed")
            return dataset
            
        except Exception as e:
            self.results['data_loading'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Data loading tests failed: {e}")
            raise
    
    def test_preprocessing(self, dataset):
        """Test data preprocessing functionality."""
        logger.info("Testing data preprocessing...")
        
        try:
            preprocessor = Preprocessor()
            
            # Test preparing for training
            processed_data = preprocessor.prepare_for_training(
                dataset['executions'],
                dataset.get('steps')
            )
            
            assert 'executions' in processed_data, "Processed executions missing"
            assert len(processed_data['executions']) > 0, "No processed executions"
            
            # Test scaling
            numeric_cols = processed_data['executions'].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaled_data = preprocessor.scale_features(processed_data['executions'])
                assert scaled_data is not None, "Feature scaling failed"
            
            # Test encoding
            if 'status' in processed_data['executions'].columns:
                encoded_data = preprocessor.encode_categorical(processed_data['executions'])
                assert encoded_data is not None, "Categorical encoding failed"
            
            self.results['preprocessing'] = {
                'status': 'PASSED',
                'processed_executions': len(processed_data['executions']),
                'numeric_features': len(numeric_cols),
                'scaling_tested': len(numeric_cols) > 0,
                'encoding_tested': True
            }
            
            logger.info("✓ Preprocessing tests passed")
            return processed_data
            
        except Exception as e:
            self.results['preprocessing'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Preprocessing tests failed: {e}")
            raise
    
    def test_feature_engineering(self, dataset):
        """Test feature engineering functionality."""
        logger.info("Testing feature engineering...")
        
        try:
            feature_engineer = FeatureEngineer()
            
            # Test workflow features
            workflow_features = feature_engineer.create_workflow_features(dataset['executions'])
            assert workflow_features is not None, "Workflow features not created"
            assert len(workflow_features) > 0, "No workflow features generated"
            
            # Test execution features
            execution_features = feature_engineer.create_execution_features(
                dataset['executions'],
                dataset.get('steps')
            )
            assert execution_features is not None, "Execution features not created"
            
            # Test prediction features
            prediction_features = feature_engineer.create_prediction_features(dataset['executions'])
            assert prediction_features is not None, "Prediction features not created"
            
            # Test clustering features
            clustering_features = feature_engineer.create_clustering_features(dataset['executions'])
            assert clustering_features is not None, "Clustering features not created"
            
            # Test bottleneck features if steps available
            bottleneck_features = None
            if 'steps' in dataset and len(dataset['steps']) > 0:
                bottleneck_features = feature_engineer.create_bottleneck_features(dataset['steps'])
                assert bottleneck_features is not None, "Bottleneck features not created"
            
            self.results['feature_engineering'] = {
                'status': 'PASSED',
                'workflow_features': len(workflow_features.columns),
                'execution_features': len(execution_features.columns),
                'prediction_features': len(prediction_features.columns),
                'clustering_features': len(clustering_features.columns),
                'bottleneck_features': len(bottleneck_features.columns) if bottleneck_features is not None else 0
            }
            
            logger.info("✓ Feature engineering tests passed")
            return {
                'workflow_features': workflow_features,
                'execution_features': execution_features,
                'prediction_features': prediction_features,
                'clustering_features': clustering_features,
                'bottleneck_features': bottleneck_features
            }
            
        except Exception as e:
            self.results['feature_engineering'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Feature engineering tests failed: {e}")
            raise
    
    def test_model_training(self, features, dataset):
        """Test model training functionality."""
        logger.info("Testing model training...")
        
        try:
            trained_models = {}
            
            # Test Process Predictor
            logger.info("  Testing Process Predictor...")
            X = features['prediction_features'].select_dtypes(include=[np.number])
            y = pd.DataFrame({
                'duration': dataset['executions']['duration'],
                'status': dataset['executions']['status']
            })
            y['success_rate'] = (y['status'] == 'completed').astype(float)
            
            predictor = ProcessPredictor('random_forest')
            predictor.fit(X, y)
            predictions = predictor.predict(X)
            assert predictions is not None, "Process predictor predictions failed"
            
            trained_models['process_predictor'] = predictor
            
            # Test Bottleneck Detector
            if features['bottleneck_features'] is not None:
                logger.info("  Testing Bottleneck Detector...")
                X_bottleneck = features['bottleneck_features'].select_dtypes(include=[np.number])
                
                detector = BottleneckDetector('isolation_forest')
                detector.fit(X_bottleneck)
                bottleneck_predictions = detector.predict(X_bottleneck)
                assert bottleneck_predictions is not None, "Bottleneck detector predictions failed"
                
                trained_models['bottleneck_detector'] = detector
            
            # Test Optimization Engine
            logger.info("  Testing Optimization Engine...")
            optimizer = OptimizationEngine('performance')
            optimizer.fit(features['workflow_features'], y)
            recommendations = optimizer.generate_recommendations(
                features['workflow_features'],
                dataset['executions'],
                features['bottleneck_features']
            )
            assert recommendations is not None, "Optimization recommendations failed"
            
            trained_models['optimization_engine'] = optimizer
            
            # Test Workflow Classifier
            logger.info("  Testing Workflow Classifier...")
            classifier = WorkflowClassifier('complexity')
            classifier.fit(features['workflow_features'])
            classifications = classifier.classify_workflows(features['workflow_features'])
            assert classifications is not None, "Workflow classification failed"
            
            trained_models['workflow_classifier'] = classifier
            
            self.results['model_training'] = {
                'status': 'PASSED',
                'models_trained': list(trained_models.keys()),
                'process_predictor_features': len(X.columns),
                'bottleneck_detector_available': features['bottleneck_features'] is not None,
                'optimization_recommendations': len(recommendations) if recommendations else 0,
                'workflow_classifications': len(classifications)
            }
            
            logger.info("✓ Model training tests passed")
            return trained_models
            
        except Exception as e:
            self.results['model_training'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Model training tests failed: {e}")
            raise
    
    def test_model_evaluation(self, models, features, dataset):
        """Test model evaluation functionality."""
        logger.info("Testing model evaluation...")
        
        try:
            metrics = ModelMetrics()
            evaluation_results = {}
            
            # Test Process Predictor evaluation
            if 'process_predictor' in models:
                logger.info("  Evaluating Process Predictor...")
                predictor = models['process_predictor']
                X = features['prediction_features'].select_dtypes(include=[np.number])
                y_duration = dataset['executions']['duration']
                y_status = (dataset['executions']['status'] == 'completed').astype(float)
                
                predictions = predictor.predict(X)
                
                # Calculate metrics
                if 'duration' in predictions.columns:
                    r2 = metrics.r2_score(y_duration, predictions['duration'])
                    mae = metrics.mean_absolute_error(y_duration, predictions['duration'])
                    evaluation_results['process_predictor'] = {'r2': r2, 'mae': mae}
                
                # Cross-validation
                cv_scores = predictor.cross_validate(X, pd.DataFrame({'duration': y_duration, 'success_rate': y_status}))
                assert cv_scores is not None, "Cross-validation failed"
            
            # Test other model evaluations
            if 'bottleneck_detector' in models:
                logger.info("  Evaluating Bottleneck Detector...")
                detector = models['bottleneck_detector']
                X_bottleneck = features['bottleneck_features'].select_dtypes(include=[np.number])
                
                analysis = detector.analyze_bottlenecks(X_bottleneck, dataset.get('steps', pd.DataFrame()))
                evaluation_results['bottleneck_detector'] = {'analysis_completed': analysis is not None}
            
            if 'optimization_engine' in models:
                logger.info("  Evaluating Optimization Engine...")
                optimizer = models['optimization_engine']
                
                optimization_metrics = optimizer.evaluate_optimization(
                    features['workflow_features'],
                    dataset['executions']
                )
                evaluation_results['optimization_engine'] = optimization_metrics
            
            if 'workflow_classifier' in models:
                logger.info("  Evaluating Workflow Classifier...")
                classifier = models['workflow_classifier']
                X_workflow = features['workflow_features'].select_dtypes(include=[np.number])
                
                cluster_results = classifier.cluster_workflows(X_workflow, n_clusters=3)
                evaluation_results['workflow_classifier'] = {'clustering_completed': cluster_results is not None}
            
            self.results['model_evaluation'] = {
                'status': 'PASSED',
                'models_evaluated': list(evaluation_results.keys()),
                'evaluation_results': evaluation_results
            }
            
            logger.info("✓ Model evaluation tests passed")
            return evaluation_results
            
        except Exception as e:
            self.results['model_evaluation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Model evaluation tests failed: {e}")
            raise
    
    def test_utilities(self, models, features, dataset):
        """Test utility functions."""
        logger.info("Testing utilities...")
        
        try:
            # Test ModelMetrics
            logger.info("  Testing ModelMetrics...")
            metrics = ModelMetrics()
            
            # Generate sample predictions for testing
            y_true = np.random.random(100)
            y_pred = y_true + np.random.normal(0, 0.1, 100)
            
            # Test regression metrics
            mse = metrics.mean_squared_error(y_true, y_pred)
            mae = metrics.mean_absolute_error(y_true, y_pred)
            r2 = metrics.r2_score(y_true, y_pred)
            
            assert mse >= 0, "MSE should be non-negative"
            assert mae >= 0, "MAE should be non-negative"
            assert r2 <= 1, "R² should be <= 1"
            
            # Test classification metrics
            y_true_class = np.random.choice([0, 1], 100)
            y_pred_class = np.random.choice([0, 1], 100)
            
            accuracy = metrics.accuracy_score(y_true_class, y_pred_class)
            precision = metrics.precision_score(y_true_class, y_pred_class)
            recall = metrics.recall_score(y_true_class, y_pred_class)
            
            assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
            assert 0 <= precision <= 1, "Precision should be between 0 and 1"
            assert 0 <= recall <= 1, "Recall should be between 0 and 1"
            
            # Test Visualizer
            logger.info("  Testing Visualizer...")
            visualizer = Visualizer()
            
            # Test feature importance plot
            feature_importance = {'feature_1': 0.5, 'feature_2': 0.3, 'feature_3': 0.2}
            fig = visualizer.plot_feature_importance(feature_importance)
            assert fig is not None, "Feature importance plot failed"
            plt.close(fig)
            
            # Test workflow timeline plot
            if 'start_time' in dataset['executions'].columns:
                fig = visualizer.plot_workflow_performance_timeline(dataset['executions'])
                assert fig is not None, "Timeline plot failed"
                plt.close(fig)
            
            # Test model performance comparison
            model_results = {
                'model_1': {'accuracy': 0.85, 'precision': 0.80},
                'model_2': {'accuracy': 0.90, 'precision': 0.85}
            }
            fig = visualizer.plot_model_performance_comparison(model_results)
            assert fig is not None, "Model comparison plot failed"
            plt.close(fig)
            
            self.results['utilities'] = {
                'status': 'PASSED',
                'metrics_tested': ['mse', 'mae', 'r2', 'accuracy', 'precision', 'recall'],
                'plots_tested': ['feature_importance', 'timeline', 'model_comparison'],
                'all_plots_generated': True
            }
            
            logger.info("✓ Utilities tests passed")
            
        except Exception as e:
            self.results['utilities'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Utilities tests failed: {e}")
            raise
    
    def test_integration(self, dataset):
        """Test end-to-end integration."""
        logger.info("Testing end-to-end integration...")
        
        try:
            # Complete pipeline test
            database_url = f"sqlite:///{self.test_db_path}"
            
            # 1. Load data
            data_loader = DataLoader(database_url)
            loaded_dataset = data_loader.create_training_dataset(lookback_days=30, min_executions=1)
            
            # 2. Preprocess
            preprocessor = Preprocessor()
            processed_data = preprocessor.prepare_for_training(
                loaded_dataset['executions'],
                loaded_dataset.get('steps')
            )
            
            # 3. Engineer features
            feature_engineer = FeatureEngineer()
            prediction_features = feature_engineer.create_prediction_features(loaded_dataset['executions'])
            
            # 4. Train model
            predictor = ProcessPredictor('random_forest')
            X = prediction_features.select_dtypes(include=[np.number])
            y = pd.DataFrame({
                'duration': loaded_dataset['executions']['duration'],
                'success_rate': (loaded_dataset['executions']['status'] == 'completed').astype(float)
            })
            predictor.fit(X, y)
            
            # 5. Make predictions
            predictions = predictor.predict(X)
            
            # 6. Evaluate
            metrics = ModelMetrics()
            if 'duration' in predictions.columns:
                r2 = metrics.r2_score(y['duration'], predictions['duration'])
            
            # 7. Save and load model
            model_path = self.temp_dir / "test_model.joblib"
            predictor.save_model(str(model_path))
            
            new_predictor = ProcessPredictor('random_forest')
            new_predictor.load_model(str(model_path))
            new_predictions = new_predictor.predict(X)
            
            # Verify predictions are consistent
            assert predictions.equals(new_predictions), "Model save/load inconsistency"
            
            self.results['integration'] = {
                'status': 'PASSED',
                'pipeline_completed': True,
                'model_saved_loaded': True,
                'predictions_consistent': True,
                'final_r2_score': r2 if 'duration' in predictions.columns else None
            }
            
            logger.info("✓ Integration tests passed")
            
        except Exception as e:
            self.results['integration'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"✗ Integration tests failed: {e}")
            raise
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("Starting comprehensive ML pipeline testing...")
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Test data loading
            dataset = self.test_data_loading()
            
            # Test preprocessing
            processed_data = self.test_preprocessing(dataset)
            
            # Test feature engineering
            features = self.test_feature_engineering(dataset)
            
            # Test model training
            models = self.test_model_training(features, dataset)
            
            # Test model evaluation
            evaluation_results = self.test_model_evaluation(models, features, dataset)
            
            # Test utilities
            self.test_utilities(models, features, dataset)
            
            # Test integration
            self.test_integration(dataset)
            
            # Generate test report
            self.generate_test_report()
            
            logger.info("🎉 All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            self.generate_test_report()
            raise
        
        finally:
            self.cleanup()
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("Generating test report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASSED' if all(
                result.get('status') == 'PASSED' 
                for result in self.results.values()
            ) else 'FAILED',
            'test_results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'passed_tests': sum(1 for r in self.results.values() if r.get('status') == 'PASSED'),
                'failed_tests': sum(1 for r in self.results.values() if r.get('status') == 'FAILED')
            }
        }
        
        # Save report
        report_path = Path(__file__).parent.parent / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("ML PIPELINE TEST REPORT")
        print("="*60)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Tests Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
        print("\nDetailed Results:")
        print("-"*40)
        
        for test_name, result in self.results.items():
            status_icon = "✓" if result.get('status') == 'PASSED' else "✗"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result.get('status', 'UNKNOWN')}")
            
            if result.get('status') == 'FAILED':
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nFull report saved to: {report_path}")
        print("="*60)
        
        logger.info(f"Test report generated: {report_path}")
    
    def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Test environment cleaned up")


def main():
    """Main testing function."""
    tester = MLPipelineTester()
    
    try:
        tester.run_all_tests()
        
        print("\n🎉 ML Pipeline testing completed successfully!")
        print("The pipeline is ready for production use.")
        
    except Exception as e:
        print(f"\n❌ ML Pipeline testing failed: {e}")
        print("Please check the test report for detailed information.")
        sys.exit(1)


if __name__ == '__main__':
    main()
