from ultralytics import YOLO


class yolo():
    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path, task="detect")

    def get_plate_number(self, image_path):
        results = self.model(image_path)
        boxes = results[0].boxes.xyxy
        boxes = boxes.cpu().numpy()

        classes = results[0].boxes.cls
        classes = classes.cpu().numpy()

        detections = [(box[0].item(), int(cls_idx)) for box, cls_idx in zip(boxes, classes)]

        sorted_classes = sorted(detections, key=lambda x: x[0], reverse=True)

        sorted_classes = [results[0].names[int(idx)] for _, idx in sorted_classes]
        return "".join(sorted_classes)