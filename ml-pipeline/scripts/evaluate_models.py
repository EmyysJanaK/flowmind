"""
Model Evaluation Script for FlowMind ML Pipeline

Evaluates trained models and generates comprehensive evaluation reports.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
import sys
import os
from typing import Dict, Any, List
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate FlowMind ML models')
    
    parser.add_argument('--models-dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--database-url', type=str, required=True,
                       help='Database connection URL for test data')
    parser.add_argument('--output-dir', type=str, default='../data/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--lookback-days', type=int, default=7,
                       help='Number of recent days for test data')
    parser.add_argument('--models', nargs='+',
                       choices=['process_predictor', 'bottleneck_detector', 'optimization_engine', 'workflow_classifier'],
                       default=['process_predictor', 'bottleneck_detector', 'optimization_engine', 'workflow_classifier'],
                       help='Models to evaluate')
    parser.add_argument('--metrics', nargs='+',
                       choices=['accuracy', 'precision', 'recall', 'f1', 'auc', 'mse', 'mae', 'r2'],
                       default=['accuracy', 'precision', 'recall', 'f1', 'mse', 'mae', 'r2'],
                       help='Evaluation metrics to compute')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive evaluation report')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    
    return parser.parse_args()


def load_test_data(database_url: str, lookback_days: int):
    """Load test data from database."""
    logger.info("Loading test data...")
    
    # Initialize data loader
    data_loader = DataLoader(database_url)
    
    # Load recent data for testing
    dataset = data_loader.create_training_dataset(
        lookback_days=lookback_days,
        min_executions=1  # Lower threshold for test data
    )
    
    if not dataset:
        raise ValueError("No test data available")
    
    logger.info(f"Loaded test dataset with {dataset['metadata']['num_executions']} executions")
    return dataset


def load_trained_models(models_dir: Path, model_types: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load trained models from disk."""
    logger.info("Loading trained models...")
    
    loaded_models = {}
    
    for model_type in model_types:
        model_dir = models_dir / model_type
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            continue
        
        loaded_models[model_type] = {}
        
        if model_type == 'process_predictor':
            # Load process predictor models
            rf_path = model_dir / 'random_forest.joblib'
            gb_path = model_dir / 'gradient_boosting.joblib'
            
            if rf_path.exists():
                rf_model = ProcessPredictor('random_forest')
                rf_model.load_model(str(rf_path))
                loaded_models[model_type]['random_forest'] = rf_model
            
            if gb_path.exists():
                gb_model = ProcessPredictor('gradient_boosting')
                gb_model.load_model(str(gb_path))
                loaded_models[model_type]['gradient_boosting'] = gb_model
        
        elif model_type == 'bottleneck_detector':
            # Load bottleneck detector models
            iso_path = model_dir / 'isolation_forest.joblib'
            cluster_path = model_dir / 'clustering.joblib'
            class_path = model_dir / 'classification.joblib'
            
            if iso_path.exists():
                iso_model = BottleneckDetector('isolation_forest')
                iso_model.load_model(str(iso_path))
                loaded_models[model_type]['isolation_forest'] = iso_model
            
            if cluster_path.exists():
                cluster_model = BottleneckDetector('clustering')
                cluster_model.load_model(str(cluster_path))
                loaded_models[model_type]['clustering'] = cluster_model
            
            if class_path.exists():
                class_model = BottleneckDetector('classification')
                class_model.load_model(str(class_path))
                loaded_models[model_type]['classification'] = class_model
        
        elif model_type == 'optimization_engine':
            # Load optimization engine models
            perf_path = model_dir / 'performance.joblib'
            
            if perf_path.exists():
                perf_model = OptimizationEngine('performance')
                perf_model.load_model(str(perf_path))
                loaded_models[model_type]['performance'] = perf_model
        
        elif model_type == 'workflow_classifier':
            # Load workflow classifier models
            complexity_path = model_dir / 'complexity.joblib'
            pattern_path = model_dir / 'pattern.joblib'
            performance_path = model_dir / 'performance.joblib'
            
            if complexity_path.exists():
                complexity_model = WorkflowClassifier('complexity')
                complexity_model.load_model(str(complexity_path))
                loaded_models[model_type]['complexity'] = complexity_model
            
            if pattern_path.exists():
                pattern_model = WorkflowClassifier('pattern')
                pattern_model.load_model(str(pattern_path))
                loaded_models[model_type]['pattern'] = pattern_model
            
            if performance_path.exists():
                perf_model = WorkflowClassifier('performance')
                perf_model.load_model(str(performance_path))
                loaded_models[model_type]['performance'] = perf_model
    
    logger.info(f"Loaded {sum(len(models) for models in loaded_models.values())} models")
    return loaded_models


def prepare_test_features(dataset: Dict, models_dir: Path):
    """Prepare test features using saved preprocessors."""
    logger.info("Preparing test features...")
    
    # Load preprocessors if available
    preprocessor_path = models_dir / 'data' / 'preprocessor.joblib'
    
    if preprocessor_path.exists():
        preprocessor = Preprocessor()
        preprocessor.load_preprocessors(str(preprocessor_path))
    else:
        preprocessor = Preprocessor()
    
    feature_engineer = FeatureEngineer()
    
    # Prepare execution data
    execution_datasets = preprocessor.prepare_for_training(
        dataset['executions'],
        dataset.get('steps')
    )
    
    # Create features
    workflow_features = feature_engineer.create_workflow_features(dataset['executions'])
    execution_features = feature_engineer.create_execution_features(
        dataset['executions'],
        dataset.get('steps')
    )
    
    bottleneck_features = None
    if 'steps' in dataset and len(dataset['steps']) > 0:
        bottleneck_features = feature_engineer.create_bottleneck_features(dataset['steps'])
    
    prediction_features = feature_engineer.create_prediction_features(dataset['executions'])
    clustering_features = feature_engineer.create_clustering_features(dataset['executions'])
    
    return {
        'preprocessed_executions': execution_datasets['executions'],
        'preprocessed_steps': execution_datasets.get('steps'),
        'workflow_features': workflow_features,
        'execution_features': execution_features,
        'bottleneck_features': bottleneck_features,
        'prediction_features': prediction_features,
        'clustering_features': clustering_features
    }


def evaluate_process_predictor(models: Dict, features: Dict, dataset: Dict, metrics: ModelMetrics) -> Dict:
    """Evaluate process predictor models."""
    logger.info("Evaluating Process Predictor models...")
    
    results = {}
    
    # Prepare test data
    X = features['prediction_features'].select_dtypes(include=[np.number])
    y_duration = dataset['executions']['duration']
    y_status = (dataset['executions']['status'] == 'completed').astype(float)
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Evaluate duration predictions
        duration_metrics = {}
        if 'duration' in predictions.columns:
            y_duration_pred = predictions['duration']
            duration_metrics = {
                'mse': metrics.mean_squared_error(y_duration, y_duration_pred),
                'mae': metrics.mean_absolute_error(y_duration, y_duration_pred),
                'r2': metrics.r2_score(y_duration, y_duration_pred),
                'mape': metrics.mean_absolute_percentage_error(y_duration, y_duration_pred)
            }
        
        # Evaluate success rate predictions
        success_metrics = {}
        if 'success_rate' in predictions.columns:
            y_success_pred = predictions['success_rate']
            success_metrics = {
                'accuracy': metrics.accuracy_score(y_status, y_success_pred > 0.5),
                'precision': metrics.precision_score(y_status, y_success_pred > 0.5),
                'recall': metrics.recall_score(y_status, y_success_pred > 0.5),
                'f1': metrics.f1_score(y_status, y_success_pred > 0.5),
                'auc': metrics.roc_auc_score(y_status, y_success_pred)
            }
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        results[model_name] = {
            'duration_metrics': duration_metrics,
            'success_metrics': success_metrics,
            'feature_importance': feature_importance,
            'predictions': predictions.to_dict('records')[:10]  # Sample predictions
        }
    
    return results


def evaluate_bottleneck_detector(models: Dict, features: Dict, dataset: Dict, metrics: ModelMetrics) -> Dict:
    """Evaluate bottleneck detector models."""
    logger.info("Evaluating Bottleneck Detector models...")
    
    results = {}
    
    if features['bottleneck_features'] is None:
        logger.warning("No bottleneck features available for evaluation")
        return results
    
    X = features['bottleneck_features'].select_dtypes(include=[np.number])
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Analyze bottlenecks
        analysis = model.analyze_bottlenecks(X, dataset.get('steps', pd.DataFrame()))
        
        # Calculate anomaly scores if available
        anomaly_metrics = {}
        if hasattr(model, 'decision_function'):
            try:
                scores = model.decision_function(X)
                anomaly_metrics = {
                    'anomaly_score_mean': np.mean(scores),
                    'anomaly_score_std': np.std(scores),
                    'outlier_ratio': np.mean(predictions == -1) if model_name == 'isolation_forest' else None
                }
            except:
                pass
        
        results[model_name] = {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'analysis': analysis,
            'anomaly_metrics': anomaly_metrics
        }
    
    return results


def evaluate_optimization_engine(models: Dict, features: Dict, dataset: Dict, metrics: ModelMetrics) -> Dict:
    """Evaluate optimization engine models."""
    logger.info("Evaluating Optimization Engine models...")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Generate recommendations
        recommendations = model.generate_recommendations(
            features['workflow_features'],
            dataset['executions'],
            features['bottleneck_features']
        )
        
        # Evaluate optimization performance
        optimization_metrics = model.evaluate_optimization(
            features['workflow_features'],
            dataset['executions']
        )
        
        results[model_name] = {
            'recommendations': recommendations,
            'optimization_metrics': optimization_metrics,
            'model_info': model.get_model_info()
        }
    
    return results


def evaluate_workflow_classifier(models: Dict, features: Dict, dataset: Dict, metrics: ModelMetrics) -> Dict:
    """Evaluate workflow classifier models."""
    logger.info("Evaluating Workflow Classifier models...")
    
    results = {}
    
    X = features['workflow_features'].select_dtypes(include=[np.number])
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Make classifications
        classifications = model.classify_workflows(X)
        
        # Perform clustering
        cluster_results = model.cluster_workflows(X, n_clusters=5)
        
        # Calculate silhouette score for clustering
        clustering_metrics = {}
        try:
            from sklearn.metrics import silhouette_score
            if 'cluster' in cluster_results:
                silhouette = silhouette_score(X, cluster_results['cluster'])
                clustering_metrics['silhouette_score'] = silhouette
        except:
            pass
        
        results[model_name] = {
            'classifications': classifications.to_dict('records'),
            'clustering': cluster_results,
            'clustering_metrics': clustering_metrics
        }
    
    return results


def generate_evaluation_report(evaluation_results: Dict, output_dir: Path):
    """Generate comprehensive evaluation report."""
    logger.info("Generating evaluation report...")
    
    report_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'summary': {},
        'detailed_results': evaluation_results
    }
    
    # Create summary statistics
    for model_type, model_results in evaluation_results.items():
        summary = {}
        
        if model_type == 'process_predictor':
            # Summarize process predictor performance
            for model_name, results in model_results.items():
                if 'duration_metrics' in results:
                    summary[f'{model_name}_duration_r2'] = results['duration_metrics'].get('r2', 0)
                    summary[f'{model_name}_duration_mae'] = results['duration_metrics'].get('mae', 0)
                
                if 'success_metrics' in results:
                    summary[f'{model_name}_success_f1'] = results['success_metrics'].get('f1', 0)
                    summary[f'{model_name}_success_auc'] = results['success_metrics'].get('auc', 0)
        
        elif model_type == 'bottleneck_detector':
            # Summarize bottleneck detector performance
            for model_name, results in model_results.items():
                if 'anomaly_metrics' in results:
                    metrics = results['anomaly_metrics']
                    if metrics.get('outlier_ratio') is not None:
                        summary[f'{model_name}_outlier_ratio'] = metrics['outlier_ratio']
        
        elif model_type == 'optimization_engine':
            # Summarize optimization engine performance
            for model_name, results in model_results.items():
                if 'optimization_metrics' in results:
                    metrics = results['optimization_metrics']
                    summary[f'{model_name}_improvement_potential'] = metrics.get('improvement_potential', 0)
        
        elif model_type == 'workflow_classifier':
            # Summarize workflow classifier performance
            for model_name, results in model_results.items():
                if 'clustering_metrics' in results:
                    metrics = results['clustering_metrics']
                    summary[f'{model_name}_silhouette_score'] = metrics.get('silhouette_score', 0)
        
        report_data['summary'][model_type] = summary
    
    # Save report
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # Generate HTML report
    html_report = generate_html_report(report_data)
    with open(output_dir / 'evaluation_report.html', 'w') as f:
        f.write(html_report)
    
    logger.info(f"Evaluation report saved to {output_dir}")


def generate_html_report(report_data: Dict) -> str:
    """Generate HTML evaluation report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FlowMind ML Pipeline Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>FlowMind ML Pipeline Evaluation Report</h1>
        <p><strong>Generated:</strong> {report_data['timestamp']}</p>
        
        <h2>Executive Summary</h2>
        <table>
            <tr><th>Model Type</th><th>Key Metrics</th><th>Status</th></tr>
    """
    
    # Add summary rows
    for model_type, summary in report_data['summary'].items():
        key_metrics = []
        for metric, value in summary.items():
            if isinstance(value, (int, float)):
                key_metrics.append(f"{metric}: {value:.4f}")
        
        status = "✓ Good" if key_metrics else "⚠ Limited Data"
        
        html += f"""
            <tr>
                <td>{model_type.replace('_', ' ').title()}</td>
                <td>{', '.join(key_metrics[:3])}</td>
                <td>{status}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Detailed Results</h2>
    """
    
    # Add detailed results for each model type
    for model_type, model_results in report_data['detailed_results'].items():
        html += f"<h3>{model_type.replace('_', ' ').title()}</h3>"
        
        for model_name, results in model_results.items():
            html += f"<h4>{model_name.replace('_', ' ').title()}</h4>"
            
            # Add metrics tables based on model type
            if model_type == 'process_predictor':
                if 'duration_metrics' in results:
                    html += "<h5>Duration Prediction Metrics</h5><table>"
                    for metric, value in results['duration_metrics'].items():
                        html += f"<tr><td>{metric.upper()}</td><td>{value:.4f}</td></tr>"
                    html += "</table>"
                
                if 'success_metrics' in results:
                    html += "<h5>Success Prediction Metrics</h5><table>"
                    for metric, value in results['success_metrics'].items():
                        html += f"<tr><td>{metric.upper()}</td><td>{value:.4f}</td></tr>"
                    html += "</table>"
            
            elif model_type == 'bottleneck_detector':
                if 'anomaly_metrics' in results:
                    html += "<h5>Anomaly Detection Metrics</h5><table>"
                    for metric, value in results['anomaly_metrics'].items():
                        if value is not None:
                            html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                    html += "</table>"
            
            elif model_type == 'workflow_classifier':
                if 'clustering_metrics' in results:
                    html += "<h5>Clustering Metrics</h5><table>"
                    for metric, value in results['clustering_metrics'].items():
                        html += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                    html += "</table>"
    
    html += """
        </body>
    </html>
    """
    
    return html


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load test data
        test_dataset = load_test_data(args.database_url, args.lookback_days)
        
        # Load trained models
        models_dir = Path(args.models_dir)
        trained_models = load_trained_models(models_dir, args.models)
        
        if not trained_models:
            raise ValueError("No trained models found")
        
        # Prepare test features
        test_features = prepare_test_features(test_dataset, models_dir)
        
        # Initialize metrics calculator
        metrics = ModelMetrics()
        
        # Evaluate models
        evaluation_results = {}
        
        if 'process_predictor' in trained_models:
            evaluation_results['process_predictor'] = evaluate_process_predictor(
                trained_models['process_predictor'],
                test_features,
                test_dataset,
                metrics
            )
        
        if 'bottleneck_detector' in trained_models:
            evaluation_results['bottleneck_detector'] = evaluate_bottleneck_detector(
                trained_models['bottleneck_detector'],
                test_features,
                test_dataset,
                metrics
            )
        
        if 'optimization_engine' in trained_models:
            evaluation_results['optimization_engine'] = evaluate_optimization_engine(
                trained_models['optimization_engine'],
                test_features,
                test_dataset,
                metrics
            )
        
        if 'workflow_classifier' in trained_models:
            evaluation_results['workflow_classifier'] = evaluate_workflow_classifier(
                trained_models['workflow_classifier'],
                test_features,
                test_dataset,
                metrics
            )
        
        # Save evaluation results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Generate comprehensive report
        if args.generate_report:
            generate_evaluation_report(evaluation_results, output_dir)
        
        # Create evaluation summary
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'test_data_summary': test_dataset['metadata'],
            'models_evaluated': list(evaluation_results.keys()),
            'output_directory': str(output_dir)
        }
        
        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for model_type, results in evaluation_results.items():
            print(f"\n{model_type.upper().replace('_', ' ')}")
            print("-" * 30)
            
            for model_name, model_results in results.items():
                print(f"\n  {model_name}:")
                
                if model_type == 'process_predictor':
                    if 'duration_metrics' in model_results:
                        r2 = model_results['duration_metrics'].get('r2', 0)
                        mae = model_results['duration_metrics'].get('mae', 0)
                        print(f"    Duration R²: {r2:.4f}")
                        print(f"    Duration MAE: {mae:.4f}")
                    
                    if 'success_metrics' in model_results:
                        f1 = model_results['success_metrics'].get('f1', 0)
                        auc = model_results['success_metrics'].get('auc', 0)
                        print(f"    Success F1: {f1:.4f}")
                        print(f"    Success AUC: {auc:.4f}")
                
                elif model_type == 'bottleneck_detector':
                    if 'anomaly_metrics' in model_results:
                        metrics_data = model_results['anomaly_metrics']
                        if metrics_data.get('outlier_ratio') is not None:
                            print(f"    Outlier Ratio: {metrics_data['outlier_ratio']:.4f}")
                
                elif model_type == 'workflow_classifier':
                    if 'clustering_metrics' in model_results:
                        silhouette = model_results['clustering_metrics'].get('silhouette_score', 0)
                        if silhouette:
                            print(f"    Silhouette Score: {silhouette:.4f}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()
