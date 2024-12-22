from plate_extraction import PlateExtraction, Verbosity
import os
import numpy as np


def compare_with_dataset():
    pe = PlateExtraction()
    pe.set_verbosity(False)
    cntNotFound = 0
    accuracy = 0
    total = 1600
    good_path = []
    accXError = 0
    accYError = 0
    accWError = 0
    accHError = 0
    for i in range(1, total):
        pe.set_image_path(f"..\\Dataset\\Vehicles\\{i:04d}.jpg")
        pe.process()
        # pe.save_plate_img(f"Results\\{i:04d}.jpg")

        plate_data = pe.get_plate_data()
        label_file = open(f"..\\Dataset\\Vehicles Labeling\\{i:04d}.txt", "r")
        label_data = label_file.read()
        label_file.close()
        label_data = label_data.strip()
        label_data = label_data.replace('\n', '')
        label_data = label_data.split(" ")
        label_data = list(map(float, label_data))

        if not plate_data:
            cntNotFound += 1
            continue
        # if the plate data is found then compare with the dataset
        print(f"Image: {i:04d}.jpg")

        # check if they over lapping and get the overlapping area
        x1, y1, w1, h1 = plate_data
        x2, y2, w2, h2 = label_data[1], label_data[2], label_data[3], label_data[4]

        # check if the two rectangles are overlapping
        if abs(x1 - x2) < 0.15 and abs(y1 - y2) < 0.15:
            accuracy += 1
            good_path.append(f"Dataset\\Vehicles\\{i:04d}.jpg")
            # pe.save_plate_img(f"Results1\\{i:04d}.jpg")
            print("Match")
        else:
            cntNotFound += 1
            print("Not Match")

        accXError += abs(x1 - x2)
        accYError += abs(y1 - y2)
        accWError += abs(x1 + w1 - (x2 + w2))
        accHError += abs(y1 + h1 - (y2+h2))
        # pe.save_plate_img(f"Results3\\{i:04d}.jpg")

    print(f"Average X Error: {accXError/total*100}%")
    print(f"Average Y Error: {accYError/total*100}%")
    print(f"Average W Error: {accWError/total*100}%")
    print(f"Average H Error: {accHError/total*100}%")
    print(f"Plates found: {accuracy/total*100}%")
    print(f"Plates not found: {cntNotFound/total*100}%")


def test():
    # open directory and then move on image by image
    pe = PlateExtraction()
    path = "Good\\good_images"
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            pe.set_image_path(os.path.join(path, filename))
            pe.process()
            pe.save_plate_img(f"Results1\\{filename}")
        else:
            continue


def pickRandomNImagesAndExtractAndShow(n):
    pe = PlateExtraction()
    pe.set_verbosity(True)
    for i in range(n):
        rand = np.random.randint(1, 250)
        pe.set_image_path(f"..\\Dataset\\Vehicles\\{rand:04d}.jpg")
        pe.set_unique_title(f"Image {rand:04d}")
        pe.process()


def get_corresponding_label_data(imgPath):
    labelPath = imgPath.replace("Vehicles", "Vehicles Labeling")
    labelPath = labelPath.replace(".jpg", ".txt")

    label_file = open(labelPath, "r")
    label_file_txt = label_file.read()
    label_file.close()
    # data is in five numbers sepearted by space
    label_data = label_file_txt.strip()
    label_data = label_data.replace('\n', '')
    label_data = label_data.split(" ")
    label_data = list(map(float, label_data))

    return label_data


def save_good_images():
    pe = PlateExtraction()
    pe.set_verbosity(False)
    dir = "Good_Images"
    labelDir = "Dataset\\Vehicles"
    for filename in os.listdir(dir):
        if filename.endswith(".jpg"):
            pe.set_image_path(os.path.join(dir, filename))
            pe.process()
            labelData = get_corresponding_label_data(
                os.path.join(labelDir, filename))
            plateData = pe.get_plate_data()
            x1, y1, w1, h1 = plateData
            x2, y2, w2, h2 = labelData[1], labelData[2], labelData[3], labelData[4]

            if abs(x1 - x2) < 0.11 and abs(y1 - y2) < 0.11:
                # cv2.imwrite(f"Good_Images\\{filename}", pe.image)
                pe.save_plate_img(f"Results1\\{filename}")
            else:
                continue


if __name__ == "__main__":
    # pe = PlateExtraction()
    # pe.set_verbosity(Verbosity.ALL_STEPS)
    # pe.set_image_path("..\\Dataset\\Vehicles\\0143.jpg")
    # pe.process()
    # pickRandomNImagesAndExtractAndShow(10)
    # pe.save_plate_img("Results\\plate.jpg")
    compare_with_dataset()
    # save_good_images()
