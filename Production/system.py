import cv2
import numpy as np
import imutils
from plate_extraction import PlateExtraction, Verbosity
from character_segmentation import CharacterSegmentation
from knn_model import compute_threshold
from skimage import feature
import joblib
from character_recognition import CharacterRecognition
import os


def construct_dataset_labels(dir, output_file="labels.txt"):
    # open output file
    output_file = open(output_file, "w", encoding="utf-8")

    # loop over all files
    last_file_number = 0
    label = ""

    for file in os.listdir(dir):
        if file.endswith('.png'):
            file_name = file.split(".")[0]
            file_number = int(file_name.split("_")[0])
            if file_number > last_file_number:
                if label != "":
                    output_file.write(f"{last_file_number} {label}\n")
                last_file_number = file_number
                label = ""
            first_char = file_name.split("-")[1]
            second_char = file_name.split("-")[2]
            if first_char == "O":
                label += second_char
            else:
                label += first_char

    output_file.write(f"{last_file_number} {label}\n")


def read_labels_file(file_path):
    labels = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            file_number, label = line.split(" ")
            labels[file_number] = label
    return labels


def process_image(image_path, knn_model, threshold):
    plate_extraction = PlateExtraction()
    plate_extraction.set_verbosity(Verbosity.QUIET)
    plate_extraction.set_image_path(image_path)

    plate_extraction.process()
    plate = plate_extraction.get_plate_image()

    character_segmentation = CharacterSegmentation(
        plate, Verbosity.QUIET)
    character_segmentation.process()

    rectangles = character_segmentation.get_rectangles()

    image = character_segmentation.get_image()
    character_recognition = CharacterRecognition(
        rectangles, image, knn_model, threshold, Verbosity.QUIET)
    plate_text = character_recognition.process()
    return plate_text


if __name__ == "__main__":
    labels = read_labels_file("labels.txt")

    knn_model = joblib.load("../model_1.pkl")
    threshold = 1.50

    result = process_image(
        "../Dataset/Vehicles/1338.jpg", knn_model, threshold)
    print(f"Result: {result}")
    print(f"Expected: {labels['1338']}")
