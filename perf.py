from YOLOv8_ONNX import YOLOv8
from YOLOv8_ONNX.datareader import DataReader
import numpy as np

model_fp32 = 'models/yolov8m.onnx'
model_quant = 'models/yolov8m.quant.onnx'

dr = DataReader("./assets", model_fp32)

model = YOLOv8(model_fp32)
quant = YOLOv8(model_quant)

time_model, outputs_model = model.benchmark(dr, viz=False)
time_quant, outputs_quant = quant.benchmark(dr, viz=False)

print(f"model: {time_model / dr.datasize} s/frame")
print(f"quant: {time_quant / dr.datasize} s/frame")

mse = [
    np.mean([np.mean((m[i] - q[i])**2) for i in range(len(m))])
    for (m, q) in zip(outputs_model, outputs_quant)
]

print("MSE:", np.mean(mse))
