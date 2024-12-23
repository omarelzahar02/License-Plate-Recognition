from ultralytics import YOLO


class Yolo():
    def __init__(self, model):
        self.model = model

    def get_plate_number(self, image_path):
        results = self.model(image_path)
        boxes = results[0].boxes.xyxy
        boxes = boxes.cpu().numpy()

        classes = results[0].boxes.cls
        classes = classes.cpu().numpy()

        detections = [(box[0].item(), int(cls_idx))
                      for box, cls_idx in zip(boxes, classes)]

        sorted_classes = sorted(detections, key=lambda x: x[0], reverse=True)

        sorted_classes = [results[0].names[int(
            idx)] for _, idx in sorted_classes]
        return " ".join(sorted_classes)


if __name__ == "__main__":
    yolo_model_path = "Yolo.onnx"
    yolo_model = YOLO(yolo_model_path, task="detect")
    yolo_predictor = Yolo(yolo_model)
    print(yolo_predictor.get_plate_number("..\\Results3\\0001.jpg"))
