import cv2
import numpy as np
import imutils
import os


class PlateExtraction:
    def __init__(self, contoursCnt=10, uniqueTitle=""):
        self.image = None
        self.gray = None
        self.thresh = None
        self.contours = None
        self.plate = None
        self.debug = False
        self.blackHat = None
        self.lightedMask = None
        self.gradX = None
        self.canny = None
        self.enhancedCanny = None
        self.contoursCnt = contoursCnt
        self.contoursRectangles = []
        self.uniqueTitle = uniqueTitle

    def set_debug(self, debug):
        self.debug = debug

    def set_unique_title(self, uniqueTitle):
        self.uniqueTitle = uniqueTitle

    def set_contours_cnt(self, contoursCnt):
        self.contoursCnt = contoursCnt

    def show_image(self, title, image, wait=False):
        if self.debug:
            cv2.imshow(self.uniqueTitle + " " + title, image)
            if wait:
                cv2.waitKey(0)

    def set_image(self, image):
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.blackHat = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4)))

    def set_image_path(self, path):
        image_colored = cv2.imread(path)
        self.set_image(image_colored)

    def get_plate_variance(self, rectangle):
        blackHat = self.blackHat
        x, y, w, h = rectangle
        rect_blackHat = blackHat[y:y+h, x:x+w]
        rect_blackHat = cv2.threshold(
            rect_blackHat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(rect_blackHat.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts_filtered = []
        filtered_rectangles = []
        cnt_vertical_letters = 0

        center = h/2 + (h/2)*0.2
        # delete very small or very large rectangle.
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w*h
            if area > 50 and area < 1000:
                # print(h/w)
                # print(f"y:{y} center:{center} y+h:{y+h}")
                if h/w >= 1 and y <= center <= y+h:
                    cnt_vertical_letters += 1
                filtered_rectangles.append((x, y, w, h))

        vertical_centers = []
        for rect in filtered_rectangles:
            x, y, w, h = rect
            vertical_centers.append(y+h/2)

        variance = np.var(vertical_centers)
        # variance from the center of the image
        vertical_centers = np.array(vertical_centers)
        variance = np.var(vertical_centers - center)
        if (len(filtered_rectangles) < 2):
            variance = 1000
        # draw rectangles
        imgCpy = rect_blackHat.copy()
        for c in filtered_rectangles:
            (x, y, w, h) = c
            rand_color = np.random.randint(0, 255, size=3).tolist()
            cv2.rectangle(imgCpy, (x, y), (x + w, y + h), rand_color, 2)
        # show_images([imgCpy], ["imgCpy"])
        # print(f"vertical letters: {cnt_vertical_letters}")
        return cnt_vertical_letters

    def process(self):
        gray = self.gray
        blackHat = self.blackHat
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 4))
        erode = cv2.erode(gray, rectKernel, iterations=2)
        dilate = cv2.dilate(erode, rectKernel, iterations=2)
        self.show_image("Dilate", dilate)

        subtracted = cv2.subtract(gray, dilate)
        self.show_image("Subtracted", subtracted)

        img_threshold = cv2.threshold(
            subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.show_image("Threshold", img_threshold)

        kernel_data = np.array([[0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)

        eroded2 = cv2.erode(img_threshold, kernel_data, iterations=1)
        dilated2 = cv2.dilate(eroded2, kernel_data, iterations=1)
        self.show_image("Dilated2", dilated2)

        kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))

        eroded6 = cv2.erode(dilated2, None, iterations=2)
        dilated6 = cv2.dilate(eroded6, None, iterations=2)

        dilated3 = cv2.dilate(dilated6, kern2, iterations=1)
        eroded3 = cv2.erode(dilated3, kern2, iterations=1)
        # # show kernel
        # dilated3 = cv2.dilate(dilated2, kern2, iterations=1)
        # eroded3 = cv2.erode(dilated3, kern2, iterations=1)
        self.show_image("Eroded3", eroded3)

        cnts = cv2.findContours(
            eroded3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        imgCpy = self.image.copy()
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            rand_color = np.random.randint(0, 255, size=3).tolist()
            cv2.rectangle(imgCpy, (x, y), (x + w, y + h), rand_color, 2)

        self.show_image("Contours", imgCpy)

        connectedComponents = cv2.connectedComponentsWithStats(
            eroded3, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = connectedComponents
        output = self.image.copy()
        # loop over the number of unique connected component labels
        label_centers = []
        imgXCenter = int(self.image.shape[1] / 2)
        imgYCenter = (int(self.image.shape[0] / 2))*1.2
        for i in range(0, numLabels):
            ecludian_distance = np.sqrt(
                (imgXCenter - centroids[i][0]) ** 2 + (imgYCenter - centroids[i][1]) ** 2)
            label_centers.append((ecludian_distance, i))

        label_centers = sorted(label_centers, key=lambda x: x[0])
        plates = []
        k = numLabels
        iterations = min(k, numLabels)
        for i in range(0, iterations):
            currIdx = label_centers[i][1]
            x = stats[currIdx, cv2.CC_STAT_LEFT]
            y = stats[currIdx, cv2.CC_STAT_TOP]
            w = stats[currIdx, cv2.CC_STAT_WIDTH]
            h = stats[currIdx, cv2.CC_STAT_HEIGHT]
            area = stats[currIdx, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[currIdx]
            aspectRatio = w / h
            # clone our original image (so we can draw on it) and then draw
            # a bounding box surrounding the connected component along with
            # a circle corresponding to the centroid
            # print(f"Area: {area} Aspect Ratio: {aspectRatio}")
            if aspectRatio > 1.70 and aspectRatio < 6 and area > 1000:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
                cnt = self.get_plate_variance((x, y, w, h))
                cost = cnt/w
                # print(cost)
                plates.append((cost, (x, y, w, h)))

        self.show_image("Output", output, True)
        plates.sort(key=lambda x: -x[0])
        if len(plates) > 0:
            self.plate = plates[0][1]
            output = self.image.copy()
            x, y, w, h = self.plate
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            self.show_image("Output", output, True)

    def save_plate_img(self, path):
        if self.plate:
            x, y, w, h = self.plate
            plate = self.image[y:y+h, x:x+w]
            # if too small then ignore
            if plate.shape[0] < 5 or plate.shape[1] < 5:
                return
            cv2.imwrite(path, plate)

    def get_plate_data(self):
        if self.plate:
            x, y, w, h = self.plate
            # divide by the width and height of the image
            x /= self.image.shape[1]
            y /= self.image.shape[0]
            w /= self.image.shape[1]
            h /= self.image.shape[0]
            return x, y, w, h
        return None


def compare_with_dataset():
    pe = PlateExtraction()
    pe.set_debug(False)
    cntNotFound = 0
    accuracy = 0
    total = 250
    good_path = []
    accXError = 0
    accYError = 0
    accWError = 0
    accHError = 0
    for i in range(1, total):
        pe.set_image_path(f"Dataset\\Vehicles\\{i:04d}.jpg")
        pe.process()
        # pe.save_plate_img(f"Results\\{i:04d}.jpg")

        plate_data = pe.get_plate_data()
        label_file = open(f"Dataset\\Vehicles Labeling\\{i:04d}.txt", "r")
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
        # print(
        #     f"Plate x: {plate_data[0]} y: {plate_data[1]} w: {plate_data[2]} h: {plate_data[3]}")
        # print(
        #     f"Label x: {label_data[1]} y: {label_data[2]} w: {label_data[3]} h: {label_data[4]}")
        # write a cost function to compute how far are the two rectangles or how they are overlapping
        # if the cost is less than a threshold then it is a match

        # check if they over lapping and get the overlapping area
        x1, y1, w1, h1 = plate_data
        x2, y2, w2, h2 = label_data[1], label_data[2], label_data[3], label_data[4]
        # check if the two rectangles are overlapping
        # where x1 within 20% of x2 and y1 within 50% of y2
        # print(f"diffX: {abs(x1 - x2)} diffY: {abs(y1 - y2)}")
        if abs(x1 - x2) < 0.11 and abs(y1 - y2) < 0.11:
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
    print(f"Accuracy: {accuracy/total*100}%")
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
    pe.set_debug(True)
    for i in range(n):
        rand = np.random.randint(1, 250)
        pe.set_image_path(f"Dataset\\Vehicles\\{rand:04d}.jpg")
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
    pe.set_debug(False)
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
    # pe.set_debug(True)
    # pe.set_image_path("Dataset\\Vehicles\\0004.jpg")
    # pe.process()
    # pickRandomNImagesAndExtractAndShow(10)
    # pe.save_plate_img("Results\\plate.jpg")
    compare_with_dataset()
    # save_good_images()
