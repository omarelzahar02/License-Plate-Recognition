import cv2
import numpy as np
import imutils
import os


def train_svm(num_samples):
    max_samples = 2000

    random_numbers = np.random.randint(1, max_samples, num_samples)
    for i in random_numbers:
        img = cv2.imread(f"Dataset\\Vehicles\\{i:04d}.jpg")

        label_file = open(f"Dataset\\Vehicles Labeling\\{i:04d}.txt", "r")
        label_data = label_file.read()
        label_file.close()
        label_data = label_data.strip()
        label_data = label_data.replace('\n', '')
        label_data = label_data.split(" ")
        label_data = list(map(float, label_data))

        x, y, w, h = label_data[1], label_data[2], label_data[3], label_data[4]
        img_height, img_width, _ = img.shape

        # x,y,w,h are percentages
        x = int(x * img_width)
        y = int(y * img_height)
        w = int(w * img_width)
        h = int(h * img_height)

        plate = img[y:y+h, x:x+w]

        cv2.imshow("Plate", plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    train_svm(5)
