"""
Model Converter Package

A simple package for converting TensorFlow models to ONNX and TensorRT formats.

Usage:
    from model_converter import ModelConverter

    converter = ModelConverter(precision='fp16')
    result = converter.convert_full_pipeline(
        tf_model_path='path/to/model',
        output_dir='outputs',
        model_name='my_model'
    )
"""

from .converter import ModelConverter

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Define what gets imported with "from model_converter import *"
__all__ = ["ModelConverter"]
