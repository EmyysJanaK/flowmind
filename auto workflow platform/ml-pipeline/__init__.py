"""
FlowMind ML Pipeline

This module contains machine learning models and pipelines for:
- Process prediction
- Bottleneck detection  
- Optimization recommendations
- Workflow classification
- Anomaly detection
"""

__version__ = "1.0.0"
__author__ = "FlowMind Team"

from .src.models import (
    ProcessPredictor,
    BottleneckDetector,
    OptimizationEngine,
    WorkflowClassifier
)

from .src.data import DataLoader, Preprocessor, FeatureEngineer
from .src.utils import ModelMetrics, Visualizer

__all__ = [
    "ProcessPredictor",
    "BottleneckDetector", 
    "OptimizationEngine",
    "WorkflowClassifier",
    "DataLoader",
    "Preprocessor",
    "FeatureEngineer",
    "ModelMetrics",
    "Visualizer"
]