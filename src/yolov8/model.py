from ultralytics import YOLO

class YOLOv8Model:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def load_model(self):
        return self.model

    def perform_inference(self, image):
        results = self.model(image)
        return results

    def process_results(self, results):
        detections = results.pandas().xyxy[0]  # Get detections in pandas DataFrame format
        return detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]  # Return relevant columns