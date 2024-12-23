# GateAccessControl
This project aims to build an LPR system that takes an image of a vehicle's license plate as input and outputs the registration number to give it an access according to system

# Flow
1. **Read images**
2. **Preprocess the image**
3. **Detect the license plate**
4. **Extract the license plate**
5. **Character Segmentation**
6. **Character Recognition**
7. **License number**

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

3. **Run the code**
```bash
python gui.py
```
