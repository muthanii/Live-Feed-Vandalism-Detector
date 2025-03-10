from setuptools import setup, find_packages

setup(
    name='yolov8-streamlit-app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Streamlit application for YOLOv8 object detection',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'streamlit',
        'torch',  # Assuming PyTorch is needed for YOLOv8
        'torchvision',  # Assuming torchvision is needed for YOLOv8
        'numpy',  # Commonly used for numerical operations
        'opencv-python',  # For image processing
        # Add any other dependencies required for YOLOv8
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)