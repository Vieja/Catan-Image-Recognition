import os
import cv2
import numpy as np
import random as rng

# INFO: Zdjęcia można przybliżać scrollem, dowolny klawisz przewija zdjęcie na kolejne

# Elements of those lists will be displayed in windows
processed_images = []
tmp_images = []
tmp_images2 = []


def loadImages():
    images = []
    main_dir = 'catan-photos'
    for directory in os.listdir(main_dir):
        for file_name in os.listdir(main_dir + '/' + directory):
            file_path = main_dir + '/' + directory + '/' + file_name
            images.append(cv2.imread(file_path))
    print('All images loaded')
    return images


def drawContourOnImage(image, contour):
    cv2.drawContours(image, [contour], -1, 255, cv2.FILLED)
    return image


def findBackground(image):
    min = np.percentile(image, 5)
    max = np.percentile(image, 95)
    # TODO: Adjust histogram using percentiles

    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    blue = [0.50, 0.65]
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(h, blue[0] * 180, 250, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(h, blue[1] * 180, 250, cv2.THRESH_BINARY_INV)
    background = cv2.bitwise_and(threshold, threshold2)
    background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))

    contours, hierarchy = cv2.findContours(background, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Find two biggest contours
    max_area_index = 0
    second_max_area_index = 0
    max_area = 0
    second_max_area = 0
    for i, cont in enumerate(contours):
        tmp_area = cv2.contourArea(cont)
        if tmp_area > max_area:
            second_max_area = max_area
            second_max_area_index = max_area_index
            max_area = tmp_area
            max_area_index = i
        elif tmp_area > second_max_area:
            second_max_area = tmp_area
            second_max_area_index = i

    # Największym konturem jest prawie zawsze cała plansza.
    # Drugim co do wielkości jest plansza z wyciętą wodą, która by nas bardziej interesowała.
    # Jednak na niektórych zdjęciach oba te kontury się zlewają w jeden, więc nie możemy zawsze brać tego mniejszego.
    # If the second biggest contour is inside the biggest one take the inside one
    if hierarchy[0][second_max_area_index][3] == max_area_index:
        best_contour = contours[second_max_area_index]
    else:
        best_contour = contours[max_area_index]
    return best_contour


def cutBackground(image, mask):
    image = cv2.bitwise_and(image, image, mask=mask)
    return image


def workOnImage(image):
    contour = findBackground(image)
    image_size = image.shape[:2]
    mask = np.zeros(image_size, dtype=np.uint8)
    contour_hull = cv2.convexHull(contour, False)
    # If the contour is not solid draw the covex hull instead
    if cv2.contourArea(contour) > 0.5 * cv2.contourArea(contour_hull):
        mask = drawContourOnImage(mask, contour)
    else:
        mask = drawContourOnImage(mask, contour_hull)
    tmp_images2.append(mask)
    image = cutBackground(image, mask)
    return image


def main():
    images = loadImages()
    for image in images:
        image = workOnImage(image)
        processed_images.append(image)
    print('All images processed')

    # Display images
    for i in range(max(len(processed_images), len(tmp_images), len(tmp_images2))):
        # Create window
        cv2.namedWindow('catan', cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow('catan2', cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow('catan3', cv2.WINDOW_GUI_NORMAL)
        # cv2.resizeWindow('catan', 1920, 1080)
        if i < len(processed_images):
            cv2.imshow('catan', processed_images[i])
        if i < len(tmp_images):
            cv2.imshow('catan2', tmp_images[i])
        if i < len(tmp_images2):
            cv2.imshow('catan3', tmp_images2[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
