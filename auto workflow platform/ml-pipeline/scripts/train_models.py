"""
Model Training Script for FlowMind ML Pipeline

Trains all ML models using the prepared data.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import sys
import os

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train FlowMind ML models')
    
    parser.add_argument('--database-url', type=str, required=True,
                       help='Database connection URL')
    parser.add_argument('--output-dir', type=str, default='../data/models',
                       help='Directory to save trained models')
    parser.add_argument('--lookback-days', type=int, default=30,
                       help='Number of days to look back for training data')
    parser.add_argument('--min-executions', type=int, default=10,
                       help='Minimum executions required per workflow')
    parser.add_argument('--models', nargs='+', 
                       choices=['process_predictor', 'bottleneck_detector', 'optimization_engine', 'workflow_classifier'],
                       default=['process_predictor', 'bottleneck_detector', 'optimization_engine', 'workflow_classifier'],
                       help='Models to train')
    parser.add_argument('--save-data', action='store_true',
                       help='Save processed datasets')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    
    return parser.parse_args()


def load_and_prepare_data(database_url: str, lookback_days: int, min_executions: int):
    """Load and prepare training data."""
    logger.info("Loading data from database...")
    
    # Initialize data loader
    data_loader = DataLoader(database_url)
    
    # Load training dataset
    dataset = data_loader.create_training_dataset(
        lookback_days=lookback_days,
        min_executions=min_executions
    )
    
    if not dataset:
        raise ValueError("No training data available")
    
    logger.info(f"Loaded dataset with {dataset['metadata']['num_executions']} executions")
    
    # Initialize preprocessor and feature engineer
    preprocessor = Preprocessor()
    feature_engineer = FeatureEngineer()
    
    # Prepare execution data
    logger.info("Preprocessing execution data...")
    execution_datasets = preprocessor.prepare_for_training(
        dataset['executions'],
        dataset.get('steps')
    )
    
    # Create advanced features
    logger.info("Creating advanced features...")
    
    # Workflow-level features
    workflow_features = feature_engineer.create_workflow_features(dataset['executions'])
    
    # Execution-level features
    execution_features = feature_engineer.create_execution_features(
        dataset['executions'],
        dataset.get('steps')
    )
    
    # Bottleneck features (if step data available)
    bottleneck_features = None
    if 'steps' in dataset and len(dataset['steps']) > 0:
        bottleneck_features = feature_engineer.create_bottleneck_features(dataset['steps'])
    
    # Prediction features
    prediction_features = feature_engineer.create_prediction_features(dataset['executions'])
    
    # Clustering features
    clustering_features = feature_engineer.create_clustering_features(dataset['executions'])
    
    prepared_data = {
        'raw_dataset': dataset,
        'preprocessed_executions': execution_datasets['executions'],
        'preprocessed_steps': execution_datasets.get('steps'),
        'workflow_features': workflow_features,
        'execution_features': execution_features,
        'bottleneck_features': bottleneck_features,
        'prediction_features': prediction_features,
        'clustering_features': clustering_features,
        'preprocessor': preprocessor,
        'feature_engineer': feature_engineer
    }
    
    return prepared_data


def train_process_predictor(data: dict, output_dir: Path):
    """Train the process predictor model."""
    logger.info("Training Process Predictor...")
    
    # Prepare features and targets
    X = data['prediction_features'].select_dtypes(include=[np.number])
    y = pd.DataFrame({
        'duration': data['raw_dataset']['executions']['duration'],
        'status': data['raw_dataset']['executions']['status']
    })
    
    # Create success rate from status
    y['success_rate'] = (y['status'] == 'completed').astype(float)
    
    # Train models
    predictor_rf = ProcessPredictor('random_forest')
    predictor_rf.fit(X, y)
    
    predictor_gb = ProcessPredictor('gradient_boosting')
    predictor_gb.fit(X, y)
    
    # Save models
    model_dir = output_dir / 'process_predictor'
    model_dir.mkdir(exist_ok=True)
    
    predictor_rf.save_model(str(model_dir / 'random_forest.joblib'))
    predictor_gb.save_model(str(model_dir / 'gradient_boosting.joblib'))
    
    # Cross-validation
    rf_cv_scores = predictor_rf.cross_validate(X, y)
    gb_cv_scores = predictor_gb.cross_validate(X, y)
    
    # Save results
    results = {
        'random_forest': {
            'cv_scores': rf_cv_scores,
            'feature_importance': predictor_rf.get_feature_importance()
        },
        'gradient_boosting': {
            'cv_scores': gb_cv_scores,
            'feature_importance': predictor_gb.get_feature_importance()
        }
    }
    
    with open(model_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Process Predictor training completed")
    return {'random_forest': predictor_rf, 'gradient_boosting': predictor_gb}


def train_bottleneck_detector(data: dict, output_dir: Path):
    """Train the bottleneck detector model."""
    logger.info("Training Bottleneck Detector...")
    
    if data['bottleneck_features'] is None:
        logger.warning("No step data available for bottleneck detection")
        return None
    
    # Prepare features
    X = data['bottleneck_features'].select_dtypes(include=[np.number])
    
    # Train models
    detector_iso = BottleneckDetector('isolation_forest')
    detector_iso.fit(X)
    
    detector_cluster = BottleneckDetector('clustering')
    detector_cluster.fit(X)
    
    detector_class = BottleneckDetector('classification')
    detector_class.fit(X)  # Will create labels automatically
    
    # Save models
    model_dir = output_dir / 'bottleneck_detector'
    model_dir.mkdir(exist_ok=True)
    
    detector_iso.save_model(str(model_dir / 'isolation_forest.joblib'))
    detector_cluster.save_model(str(model_dir / 'clustering.joblib'))
    detector_class.save_model(str(model_dir / 'classification.joblib'))
    
    # Analyze bottlenecks
    steps_df = data['raw_dataset'].get('steps', pd.DataFrame())
    if len(steps_df) > 0:
        iso_analysis = detector_iso.analyze_bottlenecks(X, steps_df)
        cluster_analysis = detector_cluster.analyze_bottlenecks(X, steps_df)
        class_analysis = detector_class.analyze_bottlenecks(X, steps_df)
        
        results = {
            'isolation_forest': iso_analysis,
            'clustering': cluster_analysis,
            'classification': class_analysis
        }
        
        with open(model_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    logger.info("Bottleneck Detector training completed")
    return {
        'isolation_forest': detector_iso,
        'clustering': detector_cluster,
        'classification': detector_class
    }


def train_optimization_engine(data: dict, output_dir: Path):
    """Train the optimization engine model."""
    logger.info("Training Optimization Engine...")
    
    # Prepare features and targets
    X = data['workflow_features'].select_dtypes(include=[np.number])
    y = data['raw_dataset']['executions'][['duration', 'success_rate']].fillna(1.0)
    
    # Train models
    optimizer_perf = OptimizationEngine('performance')
    optimizer_perf.fit(X, y)
    
    # Save model
    model_dir = output_dir / 'optimization_engine'
    model_dir.mkdir(exist_ok=True)
    
    optimizer_perf.save_model(str(model_dir / 'performance.joblib'))
    
    # Generate sample recommendations
    recommendations = optimizer_perf.generate_recommendations(
        data['workflow_features'],
        data['raw_dataset']['executions'],
        data['bottleneck_features']
    )
    
    results = {
        'sample_recommendations': recommendations,
        'model_info': optimizer_perf.get_model_info()
    }
    
    with open(model_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Optimization Engine training completed")
    return {'performance': optimizer_perf}


def train_workflow_classifier(data: dict, output_dir: Path):
    """Train the workflow classifier model."""
    logger.info("Training Workflow Classifier...")
    
    # Prepare features
    X = data['workflow_features'].select_dtypes(include=[np.number])
    
    # Train models
    classifier_complexity = WorkflowClassifier('complexity')
    classifier_complexity.fit(X)
    
    classifier_pattern = WorkflowClassifier('pattern')
    classifier_pattern.fit(X)
    
    classifier_performance = WorkflowClassifier('performance')
    classifier_performance.fit(X)
    
    # Save models
    model_dir = output_dir / 'workflow_classifier'
    model_dir.mkdir(exist_ok=True)
    
    classifier_complexity.save_model(str(model_dir / 'complexity.joblib'))
    classifier_pattern.save_model(str(model_dir / 'pattern.joblib'))
    classifier_performance.save_model(str(model_dir / 'performance.joblib'))
    
    # Generate classifications
    complexity_results = classifier_complexity.classify_workflows(X)
    pattern_results = classifier_pattern.classify_workflows(X)
    performance_results = classifier_performance.classify_workflows(X)
    
    # Clustering analysis
    cluster_results = classifier_complexity.cluster_workflows(X, n_clusters=5)
    
    results = {
        'complexity_classification': complexity_results.to_dict('records'),
        'pattern_classification': pattern_results.to_dict('records'),
        'performance_classification': performance_results.to_dict('records'),
        'clustering_analysis': cluster_results
    }
    
    with open(model_dir / 'classification_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Workflow Classifier training completed")
    return {
        'complexity': classifier_complexity,
        'pattern': classifier_pattern,
        'performance': classifier_performance
    }


def generate_visualizations(data: dict, models: dict, output_dir: Path):
    """Generate visualization plots."""
    logger.info("Generating visualizations...")
    
    visualizer = Visualizer()
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Model performance comparison (if applicable)
    model_results = {}
    
    # Process predictor performance
    if 'process_predictor' in models:
        for model_type, model in models['process_predictor'].items():
            if hasattr(model, 'model_metadata') and 'performance_metrics' in model.model_metadata:
                model_results[f'process_predictor_{model_type}'] = model.model_metadata['performance_metrics']
    
    if model_results:
        fig = visualizer.plot_model_performance_comparison(
            model_results,
            save_path=str(plots_dir / 'model_performance_comparison.png')
        )
        plt.close(fig)
    
    # Feature importance plots
    if 'process_predictor' in models:
        rf_model = models['process_predictor'].get('random_forest')
        if rf_model:
            importance = rf_model.get_feature_importance()
            if 'duration' in importance:
                fig = visualizer.plot_feature_importance(
                    importance['duration'],
                    save_path=str(plots_dir / 'duration_feature_importance.png')
                )
                plt.close(fig)
    
    # Workflow performance timeline
    if 'start_time' in data['raw_dataset']['executions'].columns:
        fig = visualizer.plot_workflow_performance_timeline(
            data['raw_dataset']['executions'],
            save_path=str(plots_dir / 'workflow_timeline.png')
        )
        plt.close(fig)
    
    # Bottleneck analysis
    if data['bottleneck_features'] is not None:
        bottleneck_data = data['bottleneck_features'].copy()
        
        # Add bottleneck predictions if detector is available
        if 'bottleneck_detector' in models:
            detector = models['bottleneck_detector'].get('isolation_forest')
            if detector:
                predictions = detector.predict(bottleneck_data.select_dtypes(include=[np.number]))
                bottleneck_data['is_bottleneck'] = predictions
        
        fig = visualizer.plot_bottleneck_analysis(
            bottleneck_data,
            save_path=str(plots_dir / 'bottleneck_analysis.png')
        )
        plt.close(fig)
    
    # Optimization recommendations
    if 'optimization_engine' in models:
        optimizer = models['optimization_engine'].get('performance')
        if optimizer:
            recommendations = optimizer.generate_recommendations(
                data['workflow_features'],
                data['raw_dataset']['executions'],
                data['bottleneck_features']
            )
            
            if recommendations:
                fig = visualizer.plot_optimization_recommendations(
                    recommendations,
                    save_path=str(plots_dir / 'optimization_recommendations.png')
                )
                plt.close(fig)
    
    # Create comprehensive dashboard
    fig = visualizer.create_dashboard(
        model_results,
        data['raw_dataset']['executions'],
        data['bottleneck_features'],
        save_path=str(plots_dir / 'dashboard.png')
    )
    plt.close(fig)
    
    logger.info(f"Visualizations saved to {plots_dir}")


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and prepare data
        logger.info("Starting data preparation...")
        data = load_and_prepare_data(
            args.database_url,
            args.lookback_days,
            args.min_executions
        )
        
        # Save data if requested
        if args.save_data:
            data_dir = output_dir / 'data'
            data_dir.mkdir(exist_ok=True)
            
            # Save preprocessed datasets
            data['preprocessor'].save_preprocessors(str(data_dir / 'preprocessor.joblib'))
            
            # Save feature datasets
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df.to_parquet(data_dir / f'{name}.parquet')
        
        # Train models
        trained_models = {}
        
        if 'process_predictor' in args.models:
            trained_models['process_predictor'] = train_process_predictor(data, output_dir)
        
        if 'bottleneck_detector' in args.models:
            trained_models['bottleneck_detector'] = train_bottleneck_detector(data, output_dir)
        
        if 'optimization_engine' in args.models:
            trained_models['optimization_engine'] = train_optimization_engine(data, output_dir)
        
        if 'workflow_classifier' in args.models:
            trained_models['workflow_classifier'] = train_workflow_classifier(data, output_dir)
        
        # Generate visualizations
        if args.generate_plots:
            generate_visualizations(data, trained_models, output_dir)
        
        # Save training summary
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'arguments': vars(args),
            'data_summary': data['raw_dataset']['metadata'],
            'models_trained': list(trained_models.keys()),
            'output_directory': str(output_dir)
        }
        
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
