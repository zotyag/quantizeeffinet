"""
Model Converter Package

A package for converting TensorFlow models to ONNX and TensorRT formats.

Usage:
    from model_converter import ModelConverter

    converter = ModelConverter()

    # Convert TensorFlow to ONNX
    converter.tf_to_onnx(
        input_model='path/to/model',
        output_path='model.onnx',
        precision='fp16'
    )

    # Convert ONNX to TensorRT
    converter.onnx_to_trt(
        input_model='model.onnx',
        engine_file_path='model.trt',
        precision='fp16'
    )

    # Or convert directly from TensorFlow to TensorRT
    converter.tf_to_trt(
        input_model='path/to/model',
        engine_file_path='model.trt',
        precision='fp16'
    )
"""

from .converter import ModelConverter

__version__ = "1.0.0"
__author__ = "Zolta Gombar"
__email__ = "g0z4@mailbox.unideb.hu"

__all__ = ["ModelConverter"]

