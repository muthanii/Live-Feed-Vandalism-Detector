import sys
import os
import pytest
from unittest.mock import MagicMock
from yolov8.model import YOLOv8Model

# Add the root directory of the project to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def mock_yolo(mocker):
    yolo = mocker.patch("yolov8.model.YOLO")
    return yolo


def test_load_model(mock_yolo):
    model_path = "fake_model_path"
    yolo_model = YOLOv8Model(model_path)
    model = yolo_model.load_model()
    mock_yolo.assert_called_once_with(model_path)
    assert model == mock_yolo.return_value


def test_perform_inference(mock_yolo):
    model_path = "fake_model_path"
    yolo_model = YOLOv8Model(model_path)
    image = b"fake_image_data"
    mock_yolo.return_value.__call__.return_value = "fake_results"
    results = yolo_model.perform_inference(image)
    mock_yolo.return_value.__call__.assert_called_once_with(image)
    assert results == "fake_results"


def test_process_results(mock_yolo):
    model_path = "fake_model_path"
    yolo_model = YOLOv8Model(model_path)
    mock_results = MagicMock()
    mock_results.pandas.return_value.xyxy = [MagicMock()]  # Mock the pandas DataFrame
    mock_results.pandas.return_value.xyxy[0] = MagicMock()
    mock_results.pandas.return_value.xyxy[0][
        ["xmin", "ymin", "xmax", "ymax", "confidence", "class"]
    ] = "fake_detections"
    detections = yolo_model.process_results(mock_results)
    assert detections == "fake_detections"
