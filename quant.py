from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    QuantFormat,
    QuantType
)
from YOLOv8_ONNX.datareader import DataReader

model_fp32 = 'models/yolov8m.onnx'
model_quant = 'models/yolov8m.quant.onnx'

# https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md

if 0:
    dr = DataReader("assets", model_fp32)
    quantized_model = quantize_static(
        model_fp32,
        model_quant,
        dr,
        quant_format= QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8
    )
else:
    quantized_model = quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        weight_type=QuantType.QUInt8
    )
