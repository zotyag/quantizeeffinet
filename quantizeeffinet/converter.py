import tensorflow as tf
import tf2onnx
import onnx
from onnxconverter_common import float16
import tensorrt as trt
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Literal
import logging
from .utils import make_model
from .onnx_to_trtengine import build_trt_engine


class ModelConverter:
    """
    Convert TensorFlow models to ONNX and TensorRT formats.

    This class provides methods to:
    - Convert TensorFlow models to ONNX format
    - Convert ONNX models to TensorRT engines
    - Run the full conversion pipeline

    Attributes:
        opset (int): ONNX opset version for compatibility
        precision (str): TensorRT precision mode ('fp32', 'fp16', 'int8')
        max_workspace_size (int): Maximum GPU memory for TensorRT (bytes)
    """

    def __init__(self, max_workspace_size: int = (1 << 30)):
        self.max_workspace_size = max_workspace_size
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('ModelConverter')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def convert_to_onnx(
            self,
            model_path: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None,
            precision: Literal['fp32', 'fp16'] = 'fp32',
            only_weigths_of_model: Literal['EfficientNetB3', 'EfficientNetB5', 'EfficientNetB6', None] = None,
            opset: int = 13
    ) -> tf.keras.Model:
        """
        Convert TensorFlow SavedModel to ONNX format.

        This method loads a TensorFlow SavedModel and converts it to ONNX
        using the tf2onnx library. The ONNX model can then be used with
        various inference frameworks or converted to TensorRT.

        Args:
            model_path: Path to TensorFlow SavedModel directory
            output_path: Path where ONNX model will be saved (e.g., 'model.onnx')
            input_signature: Optional input signature for the model

        Returns:
            Path object pointing to the saved ONNX model

        Raises:
            FileNotFoundError: If model_path doesn't exist
            RuntimeError: If conversion fails
        """

        model_path = Path(model_path)
        if output_path:
            output_path = Path(output_path)
        if not model_path.exists():
            raise FileNotFoundError(f"TensorFlow model not found: {model_path}")
        precision = precision.lower()
        if precision not in ['fp32', 'fp16']:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp32' or 'fp16', int8 is only supported in TensorRT")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Converting TensorFlow model to ONNX...")
        self.logger.info(f"  Input: {model_path}")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info(f"  Opset: {opset}")

        try:
            if only_weigths_of_model:
                model = make_model(3, base_model=only_weigths_of_model)
                model.load_weights(model_path)
            else:
                model = tf.keras.models.load_model(model_path)

            spec = (tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=opset,
                inputs_as_nchw = ["input"]  #This tells tf2onnx to convert input to NCHW
            )
            if (precision=="fp16"):
                model_fp16=float16.convert_float_to_float16(onnx_model)
                onnx_model=model_fp16
            if output_path:
                onnx.save(onnx_model, str(output_path))
                self._validate_onnx(output_path)
            else:
                self._validate_onnx(onnx_model)
            self.logger.info(f"Successfully converted to ONNX: {output_path}")
            return onnx_model

        except Exception as e:
            self.logger.error(f"ONNX conversion failed: {e}")
            raise RuntimeError(f"Failed to convert to ONNX: {e}")

    def _validate_onnx(self, onnx_path: Path) -> bool:
        """
        Validate ONNX model structure.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            True if valid, False otherwise
        """
        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            self.logger.info("ONNX model validation passed")
            return True
        except Exception as e:
            self.logger.warning(f"ONNX validation failed: {e}")
            return False



    def convert_to_trt(
            self,
            onnx_path: Union[str, Path],
            output_path: Union[str, Path],
            input_shape: Optional[Tuple] = None
    ) -> Path:
        """
        Convert ONNX model to TensorRT engine.

        This method builds an optimized TensorRT engine from an ONNX model.
        The engine is optimized for the specific GPU it's built on and
        provides maximum inference performance.

        Args:
            onnx_path: Path to ONNX model file
            output_path: Path where TensorRT engine will be saved (e.g., 'model.trt')
            input_shape: Optional tuple to override input shape (e.g., (1, 224, 224, 3))

        Returns:
            Path object pointing to the saved TensorRT engine

        Raises:
            FileNotFoundError: If onnx_path doesn't exist
            RuntimeError: If engine building fails

        Example:
            >>> converter = ModelConverter(precision='fp16')
            >>> trt_path = converter.convert_to_trt(
            ...     onnx_path='model.onnx',
            ...     output_path='model.trt'
            ... )
        """
        onnx_path = Path(onnx_path)
        output_path = Path(output_path)

        # Validate input
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Building TensorRT engine...")
        self.logger.info(f"  Input: {onnx_path}")
        self.logger.info(f"  Output: {output_path}")
        self.logger.info(f"  Precision: {self.precision}")
        self.logger.info(f"  Workspace: {self.max_workspace_size / (1 << 30):.1f}GB")

        try:
            # Create TensorRT builder
            with trt.Builder(self.trt_logger) as builder, \
                    builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                    builder.create_builder_config() as config, \
                    trt.OnnxParser(network, self.trt_logger) as parser:

                # Configure builder
                config.max_workspace_size = self.max_workspace_size

                # Set precision mode
                if self.precision == 'fp16':
                    if builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
                        self.logger.info("  FP16 mode enabled")
                    else:
                        self.logger.warning("  FP16 not supported, using FP32")

                elif self.precision == 'int8':
                    if builder.platform_has_fast_int8:
                        config.set_flag(trt.BuilderFlag.INT8)
                        self.logger.info("  INT8 mode enabled")
                    else:
                        self.logger.warning("  INT8 not supported, using FP32")

                # Parse ONNX model
                self.logger.info("  Parsing ONNX model...")
                with open(onnx_path, 'rb') as model_file:
                    if not parser.parse(model_file.read()):
                        error_msgs = []
                        for error_idx in range(parser.num_errors):
                            error_msgs.append(str(parser.get_error(error_idx)))
                        raise RuntimeError(f"ONNX parsing errors:\n" + "\n".join(error_msgs))

                # Override input shape if provided
                if input_shape:
                    self.logger.info(f"  Setting input shape: {input_shape}")
                    network.get_input(0).shape = input_shape

                # Build engine
                self.logger.info("  Building TensorRT engine (this may take a few minutes)...")
                serialized_engine = builder.build_serialized_network(network, config)

                if serialized_engine is None:
                    raise RuntimeError("Failed to build TensorRT engine")

                # Save engine
                with open(output_path, 'wb') as engine_file:
                    engine_file.write(serialized_engine)

                self.logger.info(f"✓ Successfully built TensorRT engine: {output_path}")
                return output_path

        except Exception as e:
            self.logger.error(f"✗ TensorRT engine building failed: {e}")
            raise RuntimeError(f"Failed to build TensorRT engine: {e}")

    def convert_full_pipeline(
            self,
            tf_model_path: Union[str, Path],
            output_dir: Union[str, Path],
            model_name: str = "model",
            keep_onnx: bool = True
    ) -> Dict[str, Path]:
        """
        Run full conversion pipeline: TensorFlow -> ONNX -> TensorRT.

        This is a convenience method that runs both conversion steps
        in sequence, handling intermediate files automatically.

        Args:
            tf_model_path: Path to TensorFlow SavedModel directory
            output_dir: Directory where output files will be saved
            model_name: Base name for output files (default: 'model')
            keep_onnx: Whether to keep intermediate ONNX file (default: True)

        Returns:
            Dictionary containing paths to generated files:
            - 'onnx': Path to ONNX model (if keep_onnx=True)
            - 'tensorrt': Path to TensorRT engine

        Example:
            >>> converter = ModelConverter(precision='fp16')
            >>> results = converter.convert_full_pipeline(
            ...     tf_model_path='saved_model/',
            ...     output_dir='outputs/',
            ...     model_name='resnet50'
            ... )
            >>> print(f"TensorRT engine: {results['tensorrt']}")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        onnx_path = output_dir / f"{model_name}.onnx"
        trt_path = output_dir / f"{model_name}.trt"

        self.logger.info("=" * 60)
        self.logger.info("Starting full conversion pipeline")
        self.logger.info("=" * 60)

        # Step 1: TensorFlow -> ONNX
        self.logger.info("Step 1/2: Converting TensorFlow to ONNX...")
        self.convert_to_onnx(tf_model_path, onnx_path)

        # Step 2: ONNX -> TensorRT
        self.logger.info("Step 2/2: Converting ONNX to TensorRT...")
        self.convert_to_trt(onnx_path, trt_path)

        # Clean up ONNX if requested
        results = {'tensorrt': trt_path}
        if keep_onnx:
            results['onnx'] = onnx_path
        else:
            onnx_path.unlink()
            self.logger.info(f"  Removed intermediate ONNX file")

        self.logger.info("=" * 60)
        self.logger.info("✓ Conversion pipeline completed successfully!")
        self.logger.info("=" * 60)

        return results

    def get_config(self) -> Dict:
        """
        Get current converter configuration.

        Returns:
            Dictionary with converter settings
        """
        return {
            'opset': self.opset,
            'precision': self.precision,
            'max_workspace_size': self.max_workspace_size
        }


# Allow running converter directly from command line
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python converter.py <tf_model_path> <output_dir> [model_name]")
        sys.exit(1)

    tf_path = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "model"

    converter = ModelConverter(precision='fp16')
    result = converter.convert_full_pipeline(tf_path, output_dir, model_name)

    print("\nConversion complete!")
    for key, path in result.items():
        print(f"  {key}: {path}")

