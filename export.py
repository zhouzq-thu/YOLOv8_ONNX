import cv2 as cv
import onnx
import shutil
from ultralytics import YOLO

src = cv.imread("assets/bus.jpg", cv.IMREAD_COLOR)

model_path = "yolov8n-seg.pt"
model = YOLO(model_path)

# model.export arguments
# Refs: https://docs.ultralytics.com/modes/export/#arguments
# ==============================================================================
# Key       Value           Description
# ------------------------------------------------------------------------------
# format    'torchscript'   format to export to
# imgsz     640             image size as scalar or (h, w) list, i.e. (640, 480)
# keras     False           use Keras for TF SavedModel export
# optimize  False           TorchScript: optimize for mobile
# half      False           FP16 quantization
# int8      False           INT8 quantization
# dynamic   False           ONNX/TensorRT: dynamic axes
# simplify  False           ONNX/TensorRT: simplify model
# opset     None            ONNX: opset version (optional, defaults to latest)
# workspace 4               TensorRT: workspace size (GB)
# nms       False           CoreML: add NMS
# ==============================================================================

if 1:
    f = model.export(format="onnx", opset=11)
    # add shape information to onnx model
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(f)), f)
    shutil.move(f, "models")


# model.predict arguments
# Refs: https://docs.ultralytics.com/modes/predict/#inference-arguments
# ==============================================================================
# Key           Value       Description
# ------------------------------------------------------------------------------
# source        'ultralytics/assets' source directory for images or videos
# conf          0.25        object confidence threshold for detection
# iou           0.7         intersection over union (IoU) threshold for NMS
# half          False       use half precision (FP16)
# device        None        device to run on, i.e. cuda device=0/1/2/3 or device=cpu
# show          False       show results if possible
# save          False       save images with results
# save_txt      False       save results as .txt file
# save_conf     False       save results with confidence scores
# save_crop     False       save cropped images with results
# hide_labels   False       hide labels
# hide_conf     False       hide confidence scores
# max_det 300   maximum     number of detections per image
# vid_stride    False       video frame-rate stride
# line_width    None        The line width of the bounding boxes. If None, it is scaled to the image size.
# visualize     False       visualize model features
# augment       False       apply image augmentation to prediction sources
# agnostic_nms  False       class-agnostic NMS
# retina_masks  False       use high-resolution segmentation masks
# classes None  filter      results by class, i.e. class=0, or class=[0,2,3]
# boxes         True        Show boxes in segmentation predictions
# ==============================================================================

results = model.predict(src)

# plot arguments
# Refs: https://docs.ultralytics.com/modes/predict/#plotting-results
# ==============================================================================
# Argument                      Description
# ------------------------------------------------------------------------------
# conf (bool)                   Whether to plot the detection confidence score.
# line_width (int, optional)    The line width of the bounding boxes. If None, it is scaled to the image size.
# font_size (float, optional)   The font size of the text. If None, it is scaled to the image size.
# font (str)                    The font to use for the text.
# pil (bool)                    Whether to use PIL for image plotting.
# example (str)                 An example string to display. Useful for indicating the expected format of the output.
# img (numpy.ndarray)           Plot to another image. if not, plot to original image.
# labels (bool)                 Whether to plot the label of bounding boxes.
# boxes (bool)                  Whether to plot the bounding boxes.
# masks (bool)                  Whether to plot the masks.
# probs (bool)                  Whether to plot classification probability.
# ==============================================================================

dst = results[0].plot()

cv.namedWindow("Output", cv.WINDOW_NORMAL)
cv.imshow("Output", dst)
cv.waitKey(0)
