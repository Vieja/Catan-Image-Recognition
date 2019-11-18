import os
import cv2
import numpy as np
import random as rng

# INFO: Zdjęcia można przybliżać scrollem, dowolny klawisz przewija zdjęcie na kolejne

# Elements of those lists will be displayed in windows
processed_images = []
tmp_images = []
tmp_images2 = []
tmp_images3 = []


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


def findFields(image):
    # Pionki zakrywają linie między polami, więc pomyślałem, żeby spróbować znaleźć pionki
    # i niejako w ich miejscu dorysować linie
    pieces_mask = removePieces(image)
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_gray = np.uint8(cv2.pow(image_gray/255, 1/2)*255)
    a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_gray = a.apply(image_gray)
    thresh = [170, 255]
    _, threshold = cv2.threshold(image_gray, thresh[0], 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(image_gray, thresh[1], 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(threshold, threshold2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.subtract(mask, pieces_mask)
    #tmp_images2.append(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for c in range(len(contours)):
        if cv2.contourArea(contours[c]) > 1000:
            contour_hull = cv2.convexHull(contours[c], False)
            random_color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            cv2.drawContours(image, contours, c, random_color, cv2.FILLED)
            # cv2.drawContours(image, [contour_hull], -1, random_color, cv2.FILLED)

    return image

def removePieces(image):
    # Jak na razie tylko 'proof of concept' działa jedynie dla niebieskich pionków i to też nie do końca
    image_size = image.shape[:2]
    blue_mask = np.zeros(image_size, dtype=np.uint8)
    blue_range = [0, 2]
    l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    image_gray = cv2.equalizeHist(b)
    _, threshold = cv2.threshold(image_gray, blue_range[0], 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(image_gray, blue_range[1], 255, cv2.THRESH_BINARY_INV)
    tmp = cv2.bitwise_and(threshold, threshold2)
    contours, hierarchy = cv2.findContours(tmp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i, cont in enumerate(contours):
        if cv2.contourArea(cont) > 100:
            hull = cv2.convexHull(cont)
            blue_mask = cv2.drawContours(blue_mask, [hull], -1, 255, cv2.FILLED)
    mask_all = blue_mask  # Join all masks together
    return mask_all

def removeFrame(image):
    h, w = image.shape[:2]
    if h > w:
        big = h
    else:
        big = w
    dim = (w, h)
    cropp = int (0.2 * big)
    cropped = image[cropp : h-cropp, cropp : w-cropp]
    # perform the actual resizing of the image and show it
    resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
    tmp_images2.append(resized)
    return image

def findSheep(image):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    h_new = [0.15, 0.28]
    s_new = [0.6, 1]
    v_new = [0.6, 1]
    #image[:, 2, :] = 255
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(h, h_new[0] * 180, 250, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(h, h_new[1] * 180, 250, cv2.THRESH_BINARY_INV)
    background1 = cv2.bitwise_and(threshold, threshold2)
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(h, v_new[0], 100, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(h, v_new[1] * 100, 250, cv2.THRESH_BINARY_INV)
    background2 = cv2.bitwise_and(threshold, threshold2)
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(h, v_new[0], 100, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(h, v_new[1] * 100, 250, cv2.THRESH_BINARY_INV)
    background3 = cv2.bitwise_and(threshold, threshold2)
    background = cv2.bitwise_and(background1,background2,background3)
    #background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))
    background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea)
    sort.reverse()
    i = 0
    for c in range(len(sort)):
        if cv2.contourArea(sort[c]) > 1000:
            i += 1
            cv2.drawContours(image, sort, c,  (0,255,0), cv2.FILLED)
            if i == 4:
                break
    tmp_images2.append(background)
    return background

def workOnImage(image):
    tmp_images.append(image)
    contour = findBackground(image)
    image_size = image.shape[:2]
    mask = np.zeros(image_size, dtype=np.uint8)
    contour_hull = cv2.convexHull(contour, False)
    # If the contour is not solid draw the covex hull instead
    if cv2.contourArea(contour) > 0.5 * cv2.contourArea(contour_hull):
        mask = drawContourOnImage(mask, contour)
    else:
        mask = drawContourOnImage(mask, contour_hull)
    image = cutBackground(image, mask)
    findSheep(image)
    #image = findFields(image)
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
        cv2.namedWindow('catan4', cv2.WINDOW_GUI_NORMAL)
        # cv2.resizeWindow('catan', 1920, 1080)
        if i < len(processed_images):
            cv2.imshow('catan', processed_images[i])
        if i < len(tmp_images):
            cv2.imshow('catan2', tmp_images[i])
        if i < len(tmp_images2):
            cv2.imshow('catan3', tmp_images2[i])
        if i < len(tmp_images3):
            cv2.imshow('catan4', tmp_images2[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()