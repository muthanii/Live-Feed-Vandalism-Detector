import sys
import os
import pytest
from unittest.mock import MagicMock
from .interface import main

# Add the root directory of the project to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def mock_streamlit(mocker):
    st = mocker.patch("streamlit.st")
    st.file_uploader.return_value = None
    return st


@pytest.fixture
def mock_model(mocker):
    load_model = mocker.patch("yolov8.model.load_model")
    perform_inference = mocker.patch("yolov8.model.perform_inference")
    return load_model, perform_inference


def test_main_no_file(mock_streamlit, mock_model):
    main()
    mock_streamlit.title.assert_called_once_with("YOLOv8 Streamlit App")
    mock_streamlit.file_uploader.assert_called_once_with(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )
    mock_streamlit.image.assert_not_called()
    mock_streamlit.subheader.assert_not_called()
    mock_streamlit.write.assert_not_called()


def test_main_with_file(mock_streamlit, mock_model):
    mock_streamlit.file_uploader.return_value = MagicMock()
    mock_streamlit.file_uploader.return_value.read.return_value = b"fake_image_data"
    load_model, perform_inference = mock_model
    load_model.return_value = "fake_model"
    perform_inference.return_value = "fake_results"

    main()

    mock_streamlit.image.assert_called_once_with(
        b"fake_image_data", caption="Uploaded Image", use_column_width=True
    )
    load_model.assert_called_once()
    perform_inference.assert_called_once_with("fake_model", b"fake_image_data")
    mock_streamlit.subheader.assert_called_once_with("Inference Results")
    mock_streamlit.write.assert_called_once_with("fake_results")
