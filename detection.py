from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        detections = []

        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, score, class_id = box.tolist()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": score,
                    "class_id": int(class_id)
                })

        return detections