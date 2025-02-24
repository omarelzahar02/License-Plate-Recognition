import cv2
import numpy as np
import imutils

from plate_extraction import PlateExtraction, Verbosity


class CharacterSegmentation:
    def __init__(self, image, verbosity=Verbosity.QUIET):
        self.verbosity = verbosity
        self.set_original_image(image)
        self.sharp_image = None
        self.rectangles = []
        self.uniqueTitle = "Character Segmentation"
        self.set_image(image)

    def set_image(self, image):
        self.show_image("Original Image", image)
        # convert image to rgb
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(image, (600, 400))
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def set_original_image(self, image):
        self.original_image = image.copy()
        self.original_gray = cv2.cvtColor(
            self.original_image, cv2.COLOR_BGR2GRAY)

    def set_unique_title(self, uniqueTitle):
        self.uniqueTitle = uniqueTitle

    def show_image(self, title, image, wait=False, Important=True):
        if (self.verbosity == Verbosity.DEBUG and Important) or self.verbosity == Verbosity.ALL_STEPS:
            cv2.imshow(self.uniqueTitle + " " + title, image)
            if wait or self.verbosity & Verbosity.WAIT_ON_EACH_STEP:
                cv2.waitKey(0)

    def get_image(self):
        return self.image

    def get_rectangles(self):
        return self.rectangles

    def process(self):
        threshold_image = self.preprocess()
        candidate_rectangles = self.find_and_draw_rectangles(threshold_image)
        filled_image = self.fill_rectangles(candidate_rectangles)
        self.rectangles = self.select_good_rectangles_and_draw(filled_image)

    def find_and_draw_rectangles(self, image):
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        output = self.image.copy()
        rectangles = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            rectangles.append((x, y, w, h))
            color = np.random.randint(0, 255, size=(3,)).tolist()
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        self.show_image("All Contours", output)
        return rectangles

    def fill_rectangles(self, rectangles):
        img_filled = np.zeros_like(self.image)
        for rectangle in rectangles:
            x, y, w, h = rectangle

            aspect_ratio = w / h
            area = w * h

            if 0 < aspect_ratio < 1.4 and 500 < area < 20000:
                cv2.rectangle(img_filled, (x, y), (x + w, y + h),
                              (255, 255, 255), thickness=cv2.FILLED)

        self.show_image('Filled Rectangles', img_filled)
        img_filled = cv2.morphologyEx(img_filled, cv2.MORPH_DILATE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10)), iterations=5)
        img_filled = cv2.cvtColor(img_filled, cv2.COLOR_BGR2GRAY)
        self.show_image('Enhanced Filled Rectangles', img_filled)
        return img_filled

    def find_rectangles(self, img_filled):
        contours = cv2.findContours(
            img_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        rectangles = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            rectangles.append((x, y, w, h))
        return rectangles

    def select_good_rectangles_and_draw(self, filled_image):
        candidate_rectangles = self.find_rectangles(filled_image)
        img_copy = self.image.copy()
        image_area = self.image.shape[0] * self.image.shape[1]
        img_width = self.image.shape[1]
        final_rectangles = []
        for r in candidate_rectangles:
            if r is not None:
                (x, y, w, h) = r
                aspect_ratio = w / h
                area = w * h
                if w < 0.6 * h and 1500 < area < image_area / 5 and img_width / 50 < w:
                    final_rectangles.append(r)
                    color = np.random.randint(0, 255, size=(3,)).tolist()
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 5)

        self.show_image("Final Rectangles", img_copy)

        return final_rectangles

    def mask_plate(self):
        img_threshold = cv2.threshold(
            self.original_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        img_threshold = cv2.erode(
            img_threshold, np.ones((6, 6), np.uint8), iterations=1)
        img_threshold = cv2.dilate(
            img_threshold, np.ones((1, 20), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.ones_like(self.original_gray)

        if len(contours) == 0:
            return cv2.resize(mask, (600, 400))
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        aspect_ratio = w / h

        if 2.2 < aspect_ratio < 5:
            mask = np.zeros_like(self.original_gray)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        mask = cv2.resize(mask, (600, 400))

        self.show_image("Mask", mask)

        return mask

    def preprocess(self):

        self.image = self.unsharp_mask(self.image, 10, 5)
        self.show_image("Unsharp Mask", self.image)
        img_reduced = self.reduce_colors(self.image, 8)
        # mask = self.mask_plate()
        # print(img_reduced.shape)
        # print(mask.shape)

        img_reduced_gray = cv2.cvtColor(img_reduced, cv2.COLOR_BGR2GRAY)

        # img_reduced = cv2.bitwise_and(img_reduced, mask)

        self.show_image("Reduced Image", img_reduced_gray)

        img_thresh = cv2.threshold(
            img_reduced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        img_thresh = cv2.bitwise_not(img_thresh)

        # Remove small noise
        kernel_data = np.array([[0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)
        self.show_image("Threshold Image", img_thresh)
        eroded = cv2.erode(img_thresh, kernel_data, iterations=2)
        dilated = cv2.dilate(eroded, kernel_data, iterations=1)

        self.show_image("Preprocessed Image", dilated)

        return dilated

    def reduce_colors(self, img, n):
        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = n
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        return res2

    def unsharp_mask(self, image, sigma=1.0, strength=1.5):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        # Subtract the blurred image from the original
        sharpened = cv2.addWeighted(
            image, 1.0 + strength, blurred, -strength, 0)
        return sharpened


if __name__ == '__main__':
    plate_extraction = PlateExtraction()
    plate_extraction.set_verbosity(Verbosity.WAIT_ON_EACH_STEP)
    plate_extraction.set_image_path("../Dataset/Vehicles/0016.jpg")
    plate_extraction.process()
    plate = plate_extraction.get_plate_image()

    character_segmentation = CharacterSegmentation(
        plate, Verbosity.ALL_STEPS)
    character_segmentation.process()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
