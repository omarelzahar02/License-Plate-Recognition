import cv2
import numpy as np
import imutils
from plate_extraction import PlateExtraction, Verbosity
from character_segmentation import CharacterSegmentation
from knn_model import compute_threshold
from skimage import feature
import joblib


class CharacterRecognition:
    def __init__(self, rectangles, image, knn_model, threshold, verbosity=Verbosity.QUIET):
        self.rectangles = rectangles
        self.image = image
        self.knn_model = knn_model
        self.threshold = threshold
        self.verbosity = verbosity

    def set_knn_model(self, knn_model):
        self.knn_model = knn_model

    def set_threshold(self, threshold):
        self.threshold = threshold

    def process(self):
        rectangles = sorted(self.rectangles, key=lambda x: -x[0])
        plate_text = ""
        for rectangle in rectangles:
            x, y, w, h = rectangle

            char_image = self.image[y:y+h, x:x+w]
            char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            char_image = cv2.resize(char_image, (20, 50))
            hog_img = feature.hog(char_image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
            char = self.knn_model.predict([hog_img])[0]
            distances, _ = self.knn_model.kneighbors([hog_img])
            nearest_distance = distances[0][0]
            if char == 'Outliers':
                continue

            if nearest_distance < self.threshold:
                plate_text += char

        return plate_text


if __name__ == "__main__":
    plate_extraction = PlateExtraction()
    plate_extraction.set_verbosity(Verbosity.DEBUG)
    plate_extraction.set_image_path("../Dataset/Vehicles/0039.jpg")

    plate_extraction.process()
    plate = plate_extraction.get_plate_image()

    character_segmentation = CharacterSegmentation(
        plate, Verbosity.ALL_STEPS)
    character_segmentation.process()

    rectangles = character_segmentation.get_rectangles()

    knn_model = joblib.load("../model_2.pkl")
    threshold = 1.5

    image = character_segmentation.get_image()
    cr = CharacterRecognition(
        rectangles, image, knn_model, threshold)
    text = cr.process()
    print(f"Result {text}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
