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
import pandas as pd
from collections import Counter
from yolo import Yolo
from ultralytics import YOLO


def construct_dataset_labels(dir, output_file="labels.txt"):
    # open output file
    output_file = open(output_file, "w", encoding="utf-8")

    df = pd.DataFrame(columns=["0, 1, 2, 3, 4, 5, 6, 7, 8, 9"])

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
    df = pd.read_csv(file_path, sep=" ", index_col=0)
    return df


def process_image(image_path, knn_model, threshold, verbosity=Verbosity.QUIET):
    plate_extraction = PlateExtraction()
    plate_extraction.set_verbosity(verbosity)
    plate_extraction.set_image_path(image_path)

    plate_extraction.process()
    plate = plate_extraction.get_plate_image()

    character_segmentation = CharacterSegmentation(
        plate, verbosity)
    character_segmentation.process()

    rectangles = character_segmentation.get_rectangles()

    image = character_segmentation.get_image()
    character_recognition = CharacterRecognition(
        rectangles, image, knn_model, threshold, verbosity)
    plate_text = character_recognition.process()
    return plate_text


def process_image_dnn(image_path, model, verbosity=Verbosity.QUIET):
    plate_extraction = PlateExtraction()
    plate_extraction.set_verbosity(verbosity)
    plate_extraction.set_image_path(image_path)

    plate_extraction.process()
    plate = plate_extraction.get_plate_image()
    cv2.imwrite("plate.jpg", plate)
    yolo_predictor = Yolo(model)
    # try:
    return yolo_predictor.get_plate_number("plate.jpg")
    # except:
    #     return ""


def pickRandomNImagesAndExtractAndShow(n, labels):
    pe = PlateExtraction()
    pe.set_verbosity(True)
    rand_labels = labels.sample(n)
    model = joblib.load("../model_1.pkl")
    threshold = 1.5
    for img_number in rand_labels.index:
        result = process_image(
            f"../Dataset/Vehicles/{img_number:04d}.jpg", model, threshold)
        expected = rand_labels['label'][img_number]
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        result_counter = Counter(result)
        label_counter = Counter(labels['label'][img_number])
        print(f"Testcase: {img_number}")
        if result_counter == label_counter:
            print("Correct")
        else:
            print("Incorrect")
        print("\n")


if __name__ == "__main__":
    # construct_dataset_labels("../chars_labeling/Characters")

    labels = read_labels_file("labels.txt")

    knn_model = joblib.load("../model_2.pkl")
    threshold = 1.50
    yolo_model_path = "Yolo.onnx"
    yolo_model = YOLO(yolo_model_path, task="detect")
    # print(process_image_dnn("../Dataset/Vehicles/0001.jpg", yolo_model))
    # pickRandomNImagesAndExtractAndShow(10, labels)

    correct = 0
    total = 0
    accuracy = 0

    path = "../Good/good_images"
    log_file = open("log_dnn.txt", "w", encoding="utf-8")

    for file in os.listdir(path):
        image_number = int(file.split(".")[0])
        # print(f"Processing image {image_number}")
        log_file.write(f"Processing image: {image_number}\n")
        try:
            img_path = os.path.join(path, file)
            result = process_image_dnn(
                img_path, yolo_model)

            # print(f"Result: {result}")
            # print(f"Expected: {labels['label'][image_number]}")

            total += 1

            log_file.write(f"Result: {result}\n")
            log_file.write(f"Expected: {labels['label'][image_number]}\n")

            # if result == labels['label'][image_number]:
            #     correct += 1
            result = result.replace(" ", "")
            result_counter = Counter(result)
            label_counter = Counter(labels['label'][image_number])

            # print(f"Testcase: {img_path}")
            if result_counter == label_counter:
                log_file.write("Correct\n")
                # print("Correct")
            else:
                log_file.write("Incorrect\n")
                # print("Incorrect")
            # print("\n")

            if result_counter == label_counter:
                correct += 1

        except:
            continue

    accuracy = correct / total
    log_file.write(f"Accuracy: {accuracy}\n")
    print(f"Accuracy: {accuracy}")
    log_file.close()
