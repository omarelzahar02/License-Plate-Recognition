# License Plate Recognition
This project aims to build an Plate Recognition system that takes an image of a vehicle's license plate as input and outputs the license characters.
This project is based onto Egyptian license plates.

We used classical computer vision for detection of license plates and classical machine learning for recognition of Arabic letters.

# Main Modules
1. **Plate Extraction**
2. **Character Segmentation**
3. **Character Recognition**


# Installation
1. **Clone the repository**
```bash
git clone https://github.com/omarelzahar02/GateAccessControl.git
```

2. ## Prerequisites

Make sure you have the following libraries installed:

- `customtkinter`
- `Pillow`
- `tkinterdnd2`
- `opencv-python`
- `numpy`
- `imutils`
- `scikit-image`
- `joblib`
- `pandas`
- `ultralytics`

You can install these libraries using pip:

```bash
pip install customtkinter Pillow tkinterdnd2 opencv-python numpy imutils scikit-image joblib pandas ultralytics
```

To understand the research being done and the methodology go to the notebook `Explanation.ipynb`

For production code go to Production directory where the main entry point is `system.py`. Also there is a GUI in production directory `gui.py`.

3. **Run the code**
```bash
cd Production
python gui.py
```

# Screenshots

**Input image and the plate number is the result of the recognition.**

![Screenshot](/Images/Car1.PNG)

**In the processing of the image we mark the license plate.**
![Screenshot](/Images/Car1_Marked.PNG)


**Input image and the plate number is the result of the recognition.**
![Screenshot](/Images/Car2.PNG)

**In the processing of the image we mark the license plate.**
![Screenshot](/Images/Car2_Marked.PNG)