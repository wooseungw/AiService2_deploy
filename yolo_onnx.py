from ultralytics import YOLO



model = YOLO("best.pt")
model.eval()
model.export("yolov11_best.onnx")