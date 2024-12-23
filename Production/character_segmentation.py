import cv2
import numpy as np
import imutils

from commonfunctions import show_images
from plate_extraction import PlateExtraction, Verbosity


class CharacterSegmentation:
    def __init__(self, image, verbosity=Verbosity.QUIET):
        self.verbosity = verbosity
        self.set_original_image(image)
        self.set_image(image)
        self.sharp_image = None
        self.rectangles = []
        self.uniqueTitle = "Character Segmentation"

    def set_image(self, image):
        self.image = cv2.resize(image, (600, 400))
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def set_original_image(self, image):
        self.original_image = image.copy()
        self.original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def set_unique_title(self, uniqueTitle):
        self.uniqueTitle = uniqueTitle

    def show_image(self, title, image, wait=False, Important=True):
        if (self.verbosity == Verbosity.DEBUG and Important) or self.verbosity == Verbosity.ALL_STEPS:
            cv2.imshow(self.uniqueTitle + " " + title, image)
            if wait or self.verbosity == Verbosity.WAIT_ON_EACH_STEP:
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

    # def find_and_draw_contours(self, image):
    #     contours = cv2.findContours(
    #         image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = imutils.grab_contours(contours)

    #     img_cpy = self.image.copy()

    #     for c in contours:
    #         (x, y, w, h) = cv2.boundingRect(c)
    #         rand_color = np.random.randint(0, 255, size=3).tolist()
    #         cv2.rectangle(img_cpy, (x, y), (x + w, y + h), rand_color, 2)

    #     self.show_image("All Contours", img_cpy)

    #     return contours

    def find_and_draw_rectangles(self, image):
        connectedComponents = cv2.connectedComponentsWithStats(
            image, 8, cv2.CV_32S)

        (numLabels, labels, stats, centroids) = connectedComponents
        output = self.image.copy()
        rectangles = []
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            rectangles.append((x, y, w, h))
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        self.show_image("All Contours", output)
        return rectangles

    def fill_rectangles(self, rectangles):
        img_filled = np.zeros_like(self.image)
        for rectangle in rectangles:
            x, y, w, h = rectangle
            aspect_ratio = w / h
            area = w * h
            if 0 < aspect_ratio < 1.4 and 500 < area < 50000:
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
        img_threshold = cv2.threshold(self.original_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        img_threshold = cv2.erode(img_threshold, np.ones((6, 6), np.uint8), iterations=1)
        img_threshold = cv2.dilate(img_threshold, np.ones((1, 20), np.uint8), iterations=1)

        contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No contours found.")
            return self.original_gray
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        aspect_ratio = w / h
        print(aspect_ratio)
        if 2.2 < aspect_ratio < 5:
            mask = np.zeros_like(self.original_gray)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            masked_image = cv2.bitwise_and(self.original_gray, mask)
        else:
            masked_image = self.original_gray
            print("No colored part detected. The whole image will be used.")

        masked_image = cv2.resize(masked_image, (600, 400))
        self.show_image("Masked Image", masked_image)

        return masked_image

    def preprocess(self):
        self.show_image("Plate", self.image)
        self.show_image("Gray Plate", self.gray)

        masked_image = self.mask_plate()
        # Apply unsharp mask to strength the edges
        self.sharp_image = self.unsharp_mask(masked_image, 10, 5)
        self.show_image("Sharpened", self.sharp_image)

        # Remove noise and enhance characters
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
        erode = cv2.erode(self.sharp_image, (2, 4), iterations=1)
        dilate = cv2.dilate(erode, (3, 12), iterations=1)

        dilate = cv2.bitwise_not(dilate)
        erode = cv2.erode(dilate, None, iterations=2)
        self.show_image("Characters Enhanced", erode)

        # Apply threshold to get binary image
        threshold_image = cv2.threshold(
            erode, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        self.show_image("Threshold Image", threshold_image)

        return threshold_image

    def unsharp_mask(self, image, sigma=1.0, strength=1.5):
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)

        # Subtract the blurred image from the original
        sharpened = cv2.addWeighted(
            image, 1.0 + strength, blurred, -strength, 0)
        return sharpened


if __name__ == '__main__':
    plate_extraction = PlateExtraction()
    plate_extraction.set_verbosity(Verbosity.DEBUG)
    plate_extraction.set_image_path("../Dataset/Vehicles/0001.jpg")
    plate_extraction.process()
    plate = plate_extraction.get_plate_image()

    character_segmentation = CharacterSegmentation(
        plate, Verbosity.ALL_STEPS)
    character_segmentation.process()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
