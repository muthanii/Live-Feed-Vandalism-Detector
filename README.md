# YOLOv8 Streamlit App

This project is a web application that utilizes the YOLOv8 model for object detection, with a user-friendly interface built using Streamlit. The application allows users to upload images and receive real-time object detection results.

## Project Structure

```
yolov8-streamlit-app
├── src
│   ├── app.py                # Main entry point for the Streamlit app
│   ├── yolov8
│   │   ├── __init__.py       # Package initialization for YOLOv8
│   │   └── model.py          # YOLOv8 model implementation
│   └── streamlit
│       ├── __init__.py       # Package initialization for Streamlit
│       └── interface.py       # Streamlit interface for the application
├── requirements.txt           # Project dependencies
├── setup.py                   # Packaging information
└── README.md                  # Project documentation
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/yolov8-streamlit-app.git
   cd yolov8-streamlit-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit application, execute the following command:

```
streamlit run src/app.py
```

Once the application is running, you can access it in your web browser at `http://localhost:8501`.

## YOLOv8 Model

The YOLOv8 model is implemented in `src/yolov8/model.py`. This file contains functions for loading the model, performing inference on uploaded images, and processing the results to display them in the Streamlit interface.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
