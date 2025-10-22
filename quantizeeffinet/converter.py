import warnings

import tensorflow as tf
import tf2onnx
import onnx
from onnxconverter_common import float16
import tensorrt as trt
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Literal, List
import logging
from .utils import make_model
from .onnx_to_trtengine import build_trt_engine
import os



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

    def tf_to_onnx(
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            raise FileNotFoundError(f"TensorFlow model not found: {model_path}")
        precision = precision.lower()
        if precision not in ['fp32', 'fp16']:
            raise ValueError(f"Invalid precision: {precision}. Must be 'fp32' or 'fp16', int8 is only supported in TensorRT")
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

            spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
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
            self.logger.info(f"Successfully converted to ONNX")
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



    def onnx_to_trt(
            self,
            onnx_file_path: Union[str, Path],
            engine_file_path: Optional[Union[str, Path]] = None,
            min_batch: int = 1,
            opt_batch: int = 32,
            max_batch: int = 32,
            precision: Literal['fp32', 'fp16', 'int8'] = 'fp32',
            calibration_images: Union[str, Path, List[Union[str, Path]]] = None,
            calibration_cache: Optional[Union[str, Path]] = None,
            auto_generate_engine_path: bool = False,
    ) -> Optional[trt.ICudaEngine]:
        """
        Convert ONNX model to TensorRT engine with comprehensive error handling and validation.

        Args:
            onnx_file_path: Path to the ONNX model file.
            engine_file_path: Path where the TensorRT engine will be saved.
                             If None and auto_generate_engine_path=True, auto-generates path.
            min_batch: Minimum batch size for dynamic batching.
            opt_batch: Optimal batch size for optimization.
            max_batch: Maximum batch size for dynamic batching.
            precision: Precision mode - 'fp32', 'fp16', or 'int8'.
            calibration_images: For INT8 calibration. Can be:
                               - Single directory path (str/Path) containing images
                               - Single image file path (str/Path)
                               - List of image file paths
            calibration_cache: Path to calibration cache file (for INT8).
            auto_generate_engine_path: If True, auto-generates engine path from ONNX path.

        Returns:
            TensorRT engine object if successful, None otherwise.

        Raises:
            ValueError: If input validation fails.
            FileNotFoundError: If required files are not found.
            RuntimeError: If engine build fails.
        """

        try:
            onnx_path = Path(onnx_file_path)
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
            if not onnx_path.is_file():
                raise ValueError(f"ONNX path is not a file: {onnx_path}")
            if onnx_path.suffix.lower() != '.onnx':
                self.logger.warning(f"File extension is '{onnx_path.suffix}', expected '.onnx'")
            self.logger.info(f"✓ ONNX file validated: {onnx_path}")


            if engine_file_path is None:
                if auto_generate_engine_path:
                    engine_file_path = onnx_path.with_suffix('.trt')
                    self.logger.info(f"Auto-generated engine path: {engine_file_path}")
                else:
                    self.logger.info("No engine file path provided. Engine will not be saved to disk.")
            else:
                engine_file_path = Path(engine_file_path)


            if min_batch < 1:
                raise ValueError(f"min_batch must be >= 1, got {min_batch}")
            if opt_batch < min_batch:
                raise ValueError(f"opt_batch ({opt_batch}) must be >= min_batch ({min_batch})")
            if opt_batch > max_batch:
                opt_batch = max_batch
                warnings.warn(
                    "Optimal batch size was greater than maximum batch number! Optimal batch number is reduced to match maximum batch number",
                    Warning,)
            self.logger.info(f"✓ Batch sizes validated: min={min_batch}, opt={opt_batch}, max={max_batch}")

            valid_precisions = ['fp32', 'fp16', 'int8']
            if precision.lower() not in valid_precisions:
                raise ValueError(f"Invalid precision '{precision}'. Must be one of {valid_precisions}")
            self.logger.info(f"✓ Precision mode: {precision.upper()}")


            # ============= INT8 Calibration Validation =============
            processed_calibration_images = None
            if precision.lower() == 'int8':
                if calibration_images is None and calibration_cache is None:
                    raise ValueError(
                        "INT8 precision requires either calibration_images or calibration_cache. "
                        "Please provide at least one."
                    )

                # Process calibration_images parameter
                if calibration_images is not None:
                    calib_path = Path(calibration_images) if isinstance(calibration_images, (str, Path)) else None

                    # Case 1: Directory path - collect all images
                    if calib_path and calib_path.is_dir():
                        self.logger.info(f"Scanning directory for calibration images: {calib_path}")
                        image_extensions = {'.jpg', '.jpeg', '.png'}
                        processed_calibration_images = [
                            str(img) for img in calib_path.rglob('*')
                            if img.suffix.lower() in image_extensions and img.is_file()
                        ]
                        if not processed_calibration_images:
                            raise ValueError(f"No image files found in directory: {calib_path}")
                        self.logger.info(f"✓ Found {len(processed_calibration_images)} images in directory")

                    # Case 2: Single file path
                    elif calib_path and calib_path.is_file():
                        self.logger.info(f"Using single calibration image: {calib_path}")
                        processed_calibration_images = [str(calib_path)]

                    # Case 3: List of paths
                    elif isinstance(calibration_images, (list, tuple)):
                        if len(calibration_images) == 0:
                            raise ValueError("calibration_images list is empty")

                        processed_calibration_images = [str(Path(img)) for img in calibration_images]
                        self.logger.info(f"Using {len(processed_calibration_images)} calibration images from list")

                    # Case 4: Path doesn't exist (might be created or is invalid)
                    elif calib_path:
                        raise FileNotFoundError(f"Calibration images path not found: {calib_path}")

                    else:
                        raise TypeError(
                            f"calibration_images must be a string path, Path object, or list of paths. "
                            f"Got {type(calibration_images)}"
                        )

                # Validate calibration cache if provided
                if calibration_cache is not None:
                    cache_path = Path(calibration_cache)
                    if cache_path.exists():
                        self.logger.info(f"✓ Using existing calibration cache: {cache_path}")
                    else:
                        if calibration_images is None:
                            raise FileNotFoundError(
                                f"Calibration cache not found: {cache_path} "
                                "and no calibration_images provided to generate it."
                            )
                        self.logger.info(f"Calibration cache will be created at: {cache_path}")



            self.logger.info("=" * 60)
            self.logger.info("Starting TensorRT engine build process...")
            self.logger.info("=" * 60)
            engine = build_trt_engine(
                onnx_file_path=str(onnx_path),
                engine_file_path=str(engine_file_path) if engine_file_path else None,
                min_batch=min_batch,
                opt_batch=opt_batch,
                max_batch=max_batch,
                precision=precision,
                calibration_images=processed_calibration_images,
                calibration_cache=str(calibration_cache) if calibration_cache else None,
            )



            if engine is None:
                raise RuntimeError("Engine build failed - build_trt_engine returned None")
            self.logger.info("=" * 60)
            self.logger.info("✓ ENGINE BUILD SUCCESSFUL!")
            self.logger.info("=" * 60)
            if engine_file_path:
                engine_size_mb = Path(engine_file_path).stat().st_size / (1024 * 1024)
                self.logger.info(f"Engine saved: {engine_file_path} ({engine_size_mb:.2f} MB)")
            return engine

        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Runtime error during engine build: {e}")
            raise
        except ImportError as e:
            self.logger.error(f"Import error - missing required library: {e}")
            self.logger.error("Make sure TensorRT, CUDA, and all dependencies are properly installed")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during engine build: {type(e).__name__}: {e}")
            self.logger.exception("Full traceback:")
            raise RuntimeError(f"Failed to build TensorRT engine: {e}") from e




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


