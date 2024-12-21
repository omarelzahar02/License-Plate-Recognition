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

    def set_image_path(self, path):
        image_colored = cv2.imread(path)
        self.set_image(image_colored)

    def preprocess_image(self):
        # self.image = cv2.resize(self.image, (800, 600))
        # self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pass

    def extract_blackhat(self):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self.blackHat = cv2.morphologyEx(
            self.gray, cv2.MORPH_BLACKHAT, rectKern)
        self.show_image("BlackHat", self.blackHat)

    def extract_lighted_mask(self):
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.lightedMask = cv2.morphologyEx(
            self.gray, cv2.MORPH_CLOSE, squareKern, iterations=5)
        self.lightedMask = cv2.threshold(
            self.lightedMask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.show_image("LightedMask", self.lightedMask)

    def extract_vertical_gradient_from_blackHat(self):
        gradX = cv2.Sobel(self.blackHat, ddepth=cv2.CV_32F,
                          dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.gradX = gradX
        self.show_image("GradX", self.gradX)

    def extract_canny_edge_from_blackHat(self):
        blurred_image = cv2.GaussianBlur(self.gray, (5, 5), 1)
        self.canny = cv2.Canny(blurred_image, 100, 200)
        self.show_image("Canny", self.canny)

    def extract_canny_edge_from_gradX(self):
        blurred_image = cv2.GaussianBlur(self.gradX, (5, 5), 1)
        self.canny = cv2.Canny(blurred_image, 100, 200)
        self.show_image("Canny", self.canny)

    def enhance_canny_edge_through_morphology(self):
        cannyMasked = cv2.bitwise_and(
            self.canny, self.canny, mask=self.lightedMask)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.enhancedCanny = cv2.morphologyEx(
            cannyMasked, cv2.MORPH_CLOSE, rectKern, iterations=10)
        self.enhancedCanny = cv2.bitwise_and(
            self.enhancedCanny, self.enhancedCanny, mask=self.lightedMask)
        self.show_image("Enhanced Canny", self.enhancedCanny)

    def enhance_canny_edge_through_morphology_2(self):
        grad = self.canny
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        gradBlurred = cv2.GaussianBlur(grad, (5, 5), 0)
        gradClosed = cv2.morphologyEx(gradBlurred, cv2.MORPH_CLOSE, rectKern)
        thresholdGrad = cv2.threshold(
            gradBlurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gradEnhanced = cv2.bitwise_and(
            thresholdGrad, thresholdGrad, mask=self.lightedMask)
        dilateKern = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

        # gradEnhanced = cv2.erode(gradEnhanced, None, iterations=3)
        gradEnhanced = cv2.dilate(gradEnhanced, None, iterations=12)
        # gradEnhanced = cv2.erode(gradEnhanced, None, iterations=3)
        # gradEnhanced = cv2.dilate(gradEnhanced, None, iterations=7)
        # gradEnhanced = cv2.erode(gradEnhanced, None, iterations=1)

        gradEnhanced = cv2.bitwise_and(
            gradEnhanced, gradEnhanced, mask=self.lightedMask)
        self.enhancedCanny = gradEnhanced

    def enhance_canny_edge_through_morphology_1(self):
        grad = self.gradX
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        gradBlurred = cv2.GaussianBlur(grad, (5, 5), 0)
        gradClosed = cv2.morphologyEx(gradBlurred, cv2.MORPH_CLOSE, rectKern)
        thresholdGrad = cv2.threshold(
            gradBlurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gradEnhanced = cv2.bitwise_and(
            thresholdGrad, thresholdGrad, mask=self.lightedMask)

        gradEnhanced = cv2.erode(gradEnhanced, None, iterations=3)
        gradEnhanced = cv2.dilate(gradEnhanced, None, iterations=8)
        gradEnhanced = cv2.erode(gradEnhanced, None, iterations=1)
        gradEnhanced = cv2.dilate(gradEnhanced, None, iterations=7)
        gradEnhanced = cv2.erode(gradEnhanced, None, iterations=1)
        gradEnhanced = cv2.bitwise_and(
            gradEnhanced, gradEnhanced, mask=self.lightedMask)
        self.enhancedCanny = gradEnhanced

    def extract_contours(self):
        cnts = cv2.findContours(self.enhancedCanny.copy(
        ), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[
            :self.contoursCnt]
        self.contours = cnts

    def show_contours_with_different_colors(self, contours):
        clone = self.image.copy()
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            color = np.random.randint(0, 255, size=(3,)).tolist()
            cv2.rectangle(clone, (x, y), (x + w, y + h), color, 3)
        self.show_image("Contours", clone)

    def show_rectangles_with_different_colors(self, rectangles):
        clone = self.image.copy()
        for rect in rectangles:
            if rect:
                x, y, w, h = rect
                color = np.random.randint(0, 255, size=(3,)).tolist()
                cv2.rectangle(clone, (x, y), (x + w, y + h), color, 3)
        self.show_image("Rectangles", clone)

    def compute_nearness(self, rect1, rect2):
        # get colors from original image
        # get the differnece in histograms of the two rectangles
        # get the sum of the differences
        # return the sum

        rect1_img = self.image[rect1[1]:rect1[1] +
                               rect1[3], rect1[0]:rect1[0]+rect1[2]]
        rect2_img = self.image[rect2[1]:rect2[1] +
                               rect2[3], rect2[0]:rect2[0]+rect2[2]]

        hist1 = cv2.calcHist([rect1_img], [0, 1, 2], None, [
                             8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([rect2_img], [0, 1, 2], None, [
                             8, 8, 8], [0, 256, 0, 256, 0, 256])

        # normalize the histograms to get percentage of each color
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

        return diff

    def merge_contour_triangles(self, rectangles):
        for i in range(len(rectangles)):
            for j in range(i+1, len(rectangles)):
                if rectangles[i] and rectangles[j]:
                    x1, y1, w1, h1 = rectangles[i]
                    x2, y2, w2, h2 = rectangles[j]

                    # get mid points of the rectangles
                    mid1 = (x1+w1/2, y1+h1/2)
                    mid2 = (x2+w2/2, y2+h2/2)

                    # if the rectangles are on the same line then merge but with a margin of 20 pixels
                    if abs(mid1[1] - mid2[1]) < 20:
                        # calculate the nearness of the rectangles using cost function
                        # they may be not overlapping but near to each other
                        nearness = self.compute_nearness(
                            rectangles[i], rectangles[j])
                        horizontal_distance = abs(mid1[0] - mid2[0])
                        if nearness < 2 or (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                            rectangles[i] = (min(x1, x2), min(y1, y2), max(
                                x1+w1, x2+w2)-min(x1, x2), max(y1+h1, y2+h2)-min(y1, y2))
                            rectangles[j] = None

        return rectangles

    def process_contour_rectangles(self):
        rectangles = []
        for contour in self.contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            rectangles.append((x, y, w, h))
        mergedRectangles = self.merge_contour_triangles(rectangles)
        self.contoursRectangles = mergedRectangles

    def get_plate_variance(self, rectangle):
        x, y, w, h = rectangle
        rect_blackHat = self.blackHat[y:y+h, x:x+w]
        rect_blackHat = cv2.threshold(
            rect_blackHat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(rect_blackHat.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts_filtered = []
        filtered_rectangles = []

        # delete very small or very large rectangle.
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w*h
            if area > 50 and area < 1000:
                filtered_rectangles.append((x, y, w, h))

        vertical_centers = []
        for rect in filtered_rectangles:
            x, y, w, h = rect
            vertical_centers.append(y+h/2)

        variance = np.var(vertical_centers)

        if (len(filtered_rectangles) < 2):
            variance = 1000
        return variance

    def extract_plate_from_rectangles(self):
        candidateRectangles = []
        for rect in self.contoursRectangles:
            if rect:
                x, y, w, h = rect
                aspectRatio = w / float(h)
                area = w*h
                if aspectRatio >= 1.3 and aspectRatio <= 4:
                    rectVariance = self.get_plate_variance(rect)
                    candidateRectangles.append((rectVariance, rect))

        sortedRectangles = sorted(candidateRectangles, key=lambda x: x[0])
        if len(sortedRectangles) > 0:
            self.plate = sortedRectangles[0][1]

        imgCopy = self.image.copy()
        if self.plate:
            x, y, w, h = self.plate
            cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 255, 0), 3)
        self.show_image("Plate", imgCopy, True)

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

    def process(self):
        self.preprocess_image()
        self.extract_blackhat()
        self.extract_lighted_mask()
        self.extract_vertical_gradient_from_blackHat()
        # self.extract_canny_edge_from_blackHat()
        self.extract_canny_edge_from_gradX()
        # self.enhance_canny_edge_through_morphology()
        # self.enhance_canny_edge_through_morphology_1()
        self.enhance_canny_edge_through_morphology_2()
        self.extract_contours()
        self.show_contours_with_different_colors(self.contours)
        self.process_contour_rectangles()
        self.show_rectangles_with_different_colors(self.contoursRectangles)
        self.extract_plate_from_rectangles()

    def save_plate_img(self, path):
        if self.plate:
            x, y, w, h = self.plate
            plate = self.image[y:y+h, x:x+w]
            # if too small then ignore
            if plate.shape[0] < 5 or plate.shape[1] < 5:
                return
            cv2.imwrite(path, plate)


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
        pe.save_plate_img(f"Results2\\{i:04d}.jpg")
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
    # pe.set_image_path("Dataset\\Vehicles\\0222.jpg")
    # pe.process()
    pickRandomNImagesAndExtractAndShow(10)
    # pe.save_plate_img("Results\\plate.jpg")
    # compare_with_dataset()
    # save_good_images()
