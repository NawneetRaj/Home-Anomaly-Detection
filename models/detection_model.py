from ultralytics import YOLO
import cv2

class CrowdDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Annotate the frame
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[int(cls_id)]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label text
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return boxes, classes
