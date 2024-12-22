import cv2
import numpy as np
import imutils
import os


class Verbosity:
    QUIET = 0
    DEBUG = 1
    ALL_STEPS = 2
    WAIT_ON_EACH_STEP = 3


class PlateExtraction:
    def __init__(self, contoursCnt=10, uniqueTitle="", verbosity=Verbosity.QUIET):
        self.image = None
        self.gray = None
        self.thresh = None
        self.contours = None
        self.plate = None
        self.verbosity = verbosity
        self.blackHat = None
        self.contoursCnt = contoursCnt
        self.contoursRectangles = []
        self.uniqueTitle = uniqueTitle
        self.candidates_iterator = None
        self.candidate_plates = []

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def set_unique_title(self, uniqueTitle):
        self.uniqueTitle = uniqueTitle

    def set_contours_cnt(self, contoursCnt):
        self.contoursCnt = contoursCnt

    def show_image(self, title, image, wait=False, Important=True):
        if (self.verbosity == Verbosity.DEBUG and Important) or self.verbosity & Verbosity.ALL_STEPS:
            cv2.imshow(self.uniqueTitle + " " + title, image)
            if wait or self.verbosity & Verbosity.WAIT_ON_EACH_STEP:
                cv2.waitKey(0)

    def set_image(self, image):
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.blackHat = cv2.morphologyEx(self.gray, cv2.MORPH_BLACKHAT,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4)))

    def set_image_path(self, path):
        image_colored = cv2.imread(path)
        self.set_image(image_colored)
        img_title = path.split("\\")[-1]
        self.set_unique_title(img_title)

    def get_number_of_chars_in_plate(self, rectangle):
        blackHat = self.blackHat
        x, y, w, h = rectangle

        rect_blackHat = blackHat[y:y+h, x:x+w]
        rect_blackHat = cv2.threshold(
            rect_blackHat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        contours = cv2.findContours(rect_blackHat.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        contours_filtered = []
        filtered_rectangles = []
        candidate_letters_cnt = 0

        center = h/2 + (h/2)*0.2

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            area = w*h
            # delete very small or very large rectangle.
            if area > 50 and area < 1000:
                # keep thin rectangles and rectangles near the center
                if h/w >= 1 and y <= center <= y+h:
                    candidate_letters_cnt += 1
                filtered_rectangles.append((x, y, w, h))

        imgCpy = rect_blackHat.copy()
        for c in filtered_rectangles:
            (x, y, w, h) = c
            rand_color = np.random.randint(0, 255, size=3).tolist()
            cv2.rectangle(imgCpy, (x, y), (x + w, y + h), rand_color, 2)
        self.show_image("Sub-rectangles for a candidate plate",
                        imgCpy, Important=False)

        return candidate_letters_cnt

    def get_all_candidate_rectangles(self, image):
        connectedComponents = cv2.connectedComponentsWithStats(
            image, 4, cv2.CV_32S)

        (numLabels, labels, stats, centroids) = connectedComponents

        output = self.image.copy()

        plates = []

        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            aspectRatio = w / h

            if aspectRatio > 1.70 and aspectRatio < 6 and area > 1200:
                chars_cnt = self.get_number_of_chars_in_plate((x, y, w, h))
                cost = chars_cnt/w
                plates.append((cost, (x, y, w, h)))

                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

        self.show_image("Candidate Plates", output, True)
        return plates

    def get_all_candidates_iterator(self):
        if len(self.candidate_plates) == 0:
            return iter([])

        self.candidate_plates.sort(key=lambda x: -x[0])
        for plate in self.candidate_plates:
            yield plate

    def process(self):
        morphed_image = self.apply_morphological_operations()

        self.draw_all_contours_with_random_colors(morphed_image)

        plates = self.get_all_candidate_rectangles(morphed_image)
        self.candidate_plates = plates

        self.candidates_iterator = self.get_all_candidates_iterator()

        if self.candidates_iterator:
            try:
                _, best_plate = next(self.candidates_iterator)
            except StopIteration:
                best_plate = None
            if best_plate:
                self.plate = best_plate
                output = self.image.copy()
                x, y, w, h = best_plate
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                self.show_image("Detected Plate", output, True)

    def apply_morphological_operations(self):
        gray = self.gray
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 4))

        # Step 1: Opening to erase plate region and keep non-plate region from grayscale image
        erode = cv2.erode(gray, rect_kernel, iterations=2)
        dilate = cv2.dilate(erode, rect_kernel, iterations=2)
        self.show_image("Skewed Image", dilate, Important=False)

        # Step 2: Subtract the dilated image from the grayscale image
        # where the area of plate region will be highlighted.

        subtracted = cv2.subtract(gray, dilate)
        self.show_image("Subtracted Image to Highlight Plate",
                        subtracted, Important=False)

        # Step 3: Threshold the subtracted image to get the binary image
        img_threshold = cv2.threshold(
            subtracted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        self.show_image("Threshold Image", img_threshold, Important=False)

        # Step 4: Use diamond kernel to erode the image to remove noise
        kernel_data = np.array([[0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)

        eroded2 = cv2.erode(img_threshold, kernel_data, iterations=1)
        dilated2 = cv2.dilate(eroded2, kernel_data, iterations=1)

        # Step 5: Use Opening to remove noise more
        eroded3 = cv2.erode(dilated2, None, iterations=2)
        dilated3 = cv2.dilate(eroded3, None, iterations=2)

        self.show_image("Opening to Remove Noise", dilated2)

        # Step 6: Use Closing to fill the gaps in the plate region
        kern_spanning_horizontally = cv2.getStructuringElement(
            cv2.MORPH_RECT, (10, 4))
        dilated4 = cv2.dilate(
            dilated3, kern_spanning_horizontally, iterations=1)
        eroded4 = cv2.erode(dilated4, kern_spanning_horizontally, iterations=1)

        self.show_image("Image with Plate Whitened",
                        eroded4, Important=True)
        return eroded4

    def draw_all_contours_with_random_colors(self, image):
        contours = cv2.findContours(
            image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        img_cpy = self.image.copy()

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            rand_color = np.random.randint(0, 255, size=3).tolist()
            cv2.rectangle(img_cpy, (x, y), (x + w, y + h), rand_color, 2)

        self.show_image("All Possible Contours", img_cpy)

    def save_plate_img(self, path):
        if self.plate:
            x, y, w, h = self.plate
            plate = self.image[y:y+h, x:x+w]
            # if too small then ignore
            if plate.shape[0] < 5 or plate.shape[1] < 5:
                return
            cv2.imwrite(path, plate)

    def get_plate_image(self):
        if self.plate:
            x, y, w, h = self.plate
            return self.image[y:y+h, x:x+w]
        return self.image

    def get_plate_data(self):
        if self.plate:
            x, y, w, h = self.plate
            # x -= 0.1 * w
            # y -= 0.1 * h
            # w += 20
            # h += 20
            # divide by the width and height of the image
            x /= self.image.shape[1]
            y /= self.image.shape[0]
            w /= self.image.shape[1]
            h /= self.image.shape[0]
            return x, y, w, h
        return None
