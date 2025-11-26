from ultralytics import YOLO

# Load the YOLOv11 nano model
model = YOLO("yolo11n.pt")

# Export to ONNX format with simplification
model.export(format="onnx", simplify=True, dynamic=False, imgsz=640)

print("Model exported successfully to yolo11n.onnx")
