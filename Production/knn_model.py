import cv2
import numpy as np
import imutils
from plate_extraction import PlateExtraction, Verbosity
from character_segmentation import CharacterSegmentation
import os
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage import feature
import joblib
from skimage import io


def train_knn_model(data_set_path):
    # read images
    x_train = []
    y_train = []
    for dir in os.listdir(data_set_path):

        # label is directory name
        label = dir
        for file in os.listdir(os.path.join(data_set_path, dir)):
            if file.endswith('.png'):
                char_path = os.path.join(data_set_path, dir, file)
                # check if file exists
                if not os.path.isfile(char_path):
                    continue
                try:
                    img = io.imread(char_path)
                except:
                    continue
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (20, 50))

                # extract hog features
                hog_img = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
                # hog_img = img.flatten()/255
                x_train.append(hog_img)
                y_train.append(label)

    data_set = np.array(x_train)
    labels = np.array(y_train)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        data_set, labels, test_size=0.2, random_state=1)

    # train model
    model = neighbors.KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


def compute_threshold(data_set_path, knn_model):
    # read images
    x_train = []
    y_train = []
    for dir in os.listdir(data_set_path):
        # label is directory name
        label = dir
        for file in os.listdir(os.path.join(data_set_path, dir)):
            if file.endswith('.png'):
                char_path = os.path.join(data_set_path, dir, file)
                # check if file exists
                if not os.path.isfile(char_path):
                    continue
                try:
                    img = io.imread(char_path)
                except:
                    continue
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (20, 50))

                # extract hog features
                hog_img = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
                x_train.append(hog_img)
                y_train.append(label)

    data_set = np.array(x_train)
    labels = np.array(y_train)

    distances, _ = knn_model.kneighbors(data_set)

    threshold = np.mean(distances) + 2*np.std(distances)

    return threshold
