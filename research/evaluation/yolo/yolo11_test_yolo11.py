import os
from ultralytics import YOLO

# MODEL YÜKLE
model = YOLO(os.getenv("YOLO11_MODEL_PATH", "yolo11/best.pt"))

# TEK FOTO DENEME
results = model.predict(source=os.getenv("IMAGE_PATH", "examples/sample.jpg"),
                        imgsz=640,
                        conf=0.4)

results[0].show()
