from pathlib import Path
from quantizeeffinet.quantizeeffinet import ModelConverter



if __name__ == "__main__":
    converter = ModelConverter()

    # tfpath=Path("../../csomag6_GombarZolta_EfficientNet/models_tensorflow/Full_models/tf_full_model_b6.keras")
    # onnxpath = Path("../../csomag6_GombarZolta_EfficientNet/PackageTest/model_b6_fp32_olyweigts.onnx")
    # model=converter.tf_to_onnx(tfpath,onnxpath,precision="fp32")
    #
    # print([(i.name, [d.dim_value if d.dim_value > 0 else d.dim_param for d in i.type.tensor_type.shape.dim]) for i in
    #        model.graph.input])

    onnxpath=Path("../../csomag6_GombarZolta_EfficientNet/PackageTest/model_b6_fp32_olyweigts.onnx")
    trtpath = Path("../../csomag6_GombarZolta_EfficientNet/PackageTest/int8_model_b6.engine")
    imagepath=Path("../../csomag6_GombarZolta_EfficientNet/Calibration_images/equal_2200")
    cachepath=Path("../../csomag6_GombarZolta_EfficientNet/PackageTest/b6.cache")
    model=converter.onnx_to_trt(
        onnxpath,
        trtpath,
        1,
        32,
        32,
        "int8",
        calibration_images=imagepath,
        calibration_cache=None
    )

    model=converter.onnx_to_trt(
        input_model=onnxpath,
        engine_file_path=trtpath,
        min_batch=1,
        max_batch=32,
        opt_batch=32,
        precision="int8",
        calibration_images=imagepath,
        calibration_cache=cachepath,
    )

    # tfpath = Path("../../csomag6_GombarZolta_EfficientNet/models_tensorflow/Full_models/tf_full_model_b6.keras")
    # trtpath = Path("../../csomag6_GombarZolta_EfficientNet/PackageTest/int8_model_b6.engine")
    # imagepath = Path("../../csomag6_GombarZolta_EfficientNet/Calibration_images/equal_2200")
    # cachepath = Path("../../csomag6_GombarZolta_EfficientNet/PackageTest/b6.cache")
    # model = converter.tf_to_trt(
    #     tfpath,
    #     trtpath,
    #     "EfficientNetB6",
    #     1,32,32,
    #     "int8",
    #     calibration_images=imagepath,
    #     calibration_cache=None,
    # )


