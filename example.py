import cv2
from YOLOv8_ONNX import YOLOv8, YOLOv8Seg, YOLOv8Pose

yolo = YOLOv8("models/yolov8n.onnx", conf_thres=0.3, iou_thres=0.5)
# yolo = YOLOv8Seg("models/yolov8n-seg.onnx", conf_thres=0.3, iou_thres=0.5)
# yolo = YOLOv8Pose("models/yolov8n-pose.onnx", conf_thres=0.3, iou_thres=0.5)

img = cv2.imread("assets/bus.jpg", cv2.IMREAD_COLOR)

# Detect Objects
yolo(img)

# Draw detections
combined_img = yolo.draw_results(img)
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", combined_img)
cv2.waitKey(0)
