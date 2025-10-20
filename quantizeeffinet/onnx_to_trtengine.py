import warnings
from pathlib import Path
from typing import Union, List, Optional
import tensorrt as trt
import numpy as np
import os
from cuda import cuda, cudart
from PIL import Image



class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator for TensorRT
    """
    def __init__(self,
                 calibration_images: Union[str, Path, List[Union[str, Path]]],
                 cache_file: Optional[Union[str, Path]]=None,
                 batch_size=8,
                 input_shape=(1, 3, 224, 224),
                 max_nom_of_calibration_images=10000
                 ):
        """
        Args:
            calibration_images: List of image paths for calibration
            cache_file: Path to save/load calibration cache
            batch_size: Batch size for calibration (must be fixed)
            input_shape: Input tensor shape (batch, channels, height, width)
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = None
        self._allocate_memory()

        image_list: List[Union[str, Path]] = []
        if isinstance(calibration_images, (str, Path)):
            directory_path = Path(calibration_images)
            if directory_path.is_dir():
                for ext in ('*.jpg', '*.jpeg', '*.png'):
                    image_list.extend(directory_path.glob(ext))
            else:
                image_list = [Path(calibration_images)]
        elif isinstance(calibration_images, list):
            image_list = calibration_images
        else:
            raise ValueError("calibration_images must be a directory path or a list of image paths.")
        self.calibration_images = image_list[:max_nom_of_calibration_images]

        self.input_shape = input_shape
        if len(self.input_shape) != 4:
            raise ValueError("Input shape must be 4-dimensional (N, C, H, W).")

    def _allocate_memory(self):
        batch_elements = self.batch_size * self.input_shape[1] * \
                        self.input_shape[2] * self.input_shape[3]
        buffer_size = batch_elements * np.dtype(np.float32).itemsize

        err, self.device_input = cudart.cudaMalloc(buffer_size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA malloc failed: {err}")
        print(f"Allocated {buffer_size} bytes for calibration")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """
        Get next batch of calibration data
        Returns device memory pointers
        """
        if self.current_index >= len(self.calibration_images):
            print(f"Calibration complete: processed {self.current_index} images")
            return None

        batch_imgs = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.calibration_images):
                break

            img_path = self.calibration_images[self.current_index]
            try:
                image = Image.open(img_path).convert("RGB")
                image = image.resize((self.input_shape[3], self.input_shape[2]))
                image = np.array(image, dtype=np.float32)
                img = np.transpose(image, (2, 0, 1))    # HWC to CHW
                batch_imgs.append(img)
                if (self.current_index + 1) % 100 == 0:
                    print(f"Calibrating: {self.current_index + 1}/{len(self.calibration_images)}")
                self.current_index += 1
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                self.current_index += 1
                continue

        if not batch_imgs:
            return None

        batch_data = np.ascontiguousarray(batch_imgs, dtype=np.float32)
        err, = cudart.cudaMemcpy(
            self.device_input,
            batch_data.ctypes.data,
            batch_data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA memcpy failed: {err}")

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Using cached calibration data from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Calibration cache saved to {self.cache_file}")

    def __del__(self):
        """Free device memory"""
        if self.device_input is not None:
            cudart.cudaFree(self.device_input)

def build_int8_engine(onnx_file_path,
                      calibration_images,
                      engine_file_path=None,
                      calibration_cache=None,
                      min_batch=1,
                      opt_batch=32,
                      max_batch=32,
                      input_shape=(1, 3, 224, 224)):
    """
    Build INT8 TensorRT engine from ONNX model with dynamic batch size support

    Args:
        onnx_file_path: Path to ONNX model
        engine_file_path: Path to save TensorRT engine
        calibration_images: List of image paths for INT8 calibration
        calibration_cache: Path to calibration cache file
        min_batch: Minimum batch size for dynamic batching
        opt_batch: Optimal batch size for auto-tuner optimization
        max_batch: Maximum batch size for dynamic batching
        input_shape: Input tensor shape (batch, channels, height, width)
                    Note: batch dimension will be overridden by min/opt/max_batch

    Returns:
        TensorRT engine or None if failed
    """

    try:
        err, device_count = cudart.cudaGetDeviceCount()
        if err != cudart.cudaError_t.cudaSuccess or device_count == 0:
            raise RuntimeError(f"No CUDA devices available: {err}")
        print(f"Found {device_count} CUDA device(s)")
    except Exception as e:
        print(f"CUDA initialization error: {e}")
        return None

    if opt_batch>max_batch:
        opt_batch=max_batch
        warnings.warn(
            "Optimal batch size was greater than maximum batch number! Optimal batch number is Reduced to max batch number",
            Warning
        )

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Loading ONNX file: {onnx_file_path}")
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")
    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print(f"Completed parsing ONNX file")
    print(f"Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")


    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_tensor_shape = input_tensor.shape
    print(f"Input tensor shape: {input_tensor_shape}")
    print(f"Creating optimization profile for dynamic batch sizes: min={min_batch}, opt={opt_batch}, max={max_batch}")

    profile = builder.create_optimization_profile()
    _, C, H, W = input_shape
    min_shape = (min_batch, C, H, W)
    opt_shape = (opt_batch, C, H, W)
    max_shape = (max_batch, C, H, W)
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    config.set_flag(trt.BuilderFlag.INT8)
    # For TensorRT 10.1+, also enable FP16 as a fallback
    config.set_flag(trt.BuilderFlag.FP16)

    calibrator = Int8Calibrator(
        calibration_images=calibration_images,
        cache_file=calibration_cache,
        batch_size=opt_batch,
        input_shape=(opt_batch, C, H, W)
    )

    config.int8_calibrator = calibrator

    print("Building INT8 TensorRT engine with dynamic batch support... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("ERROR: Failed to build engine")
    print("Completed building engine")

    if engine_file_path:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
            print(f"Engine saved to: {engine_file_path}")

    print(f"Engine supports dynamic batch sizes from {min_batch} to {max_batch}")
    # runtime = trt.Runtime(TRT_LOGGER)
    # engine = runtime.deserialize_cuda_engine(serialized_engine)
    # return engine

    return True