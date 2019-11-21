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


def findBackground(image, blue):
    min = np.percentile(image, 5)
    max = np.percentile(image, 95)
    # TODO: Adjust histogram using percentiles

    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
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


def findTerrain(rawData, data, color, h_new, s_new, v_new, ile, ktory):
    h, s, v = cv2.split(cv2.cvtColor(data, cv2.COLOR_BGR2HSV))
    #if (ktory == 3):
        #data[:, 2, :] = 255
    if ktory == 3:
        # Finding two thresholds and then finding the common part
        _, threshold = cv2.threshold(h, h_new[0] * 180, 180, cv2.THRESH_BINARY_INV)
        _, threshold2 = cv2.threshold(h, h_new[1] * 180, 180, cv2.THRESH_BINARY)
        background1 = cv2.bitwise_xor(threshold, threshold2)
    else:
        # Finding two thresholds and then finding the common part
        _, threshold = cv2.threshold(h, h_new[0] * 180, 180, cv2.THRESH_BINARY)
        _, threshold2 = cv2.threshold(h, h_new[1] * 180, 180, cv2.THRESH_BINARY_INV)
        background1 = cv2.bitwise_and(threshold, threshold2)
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(s, s_new[0] * 255, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(s, s_new[1] * 255, 255, cv2.THRESH_BINARY_INV)
    background2 = cv2.bitwise_and(threshold, threshold2)
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(v, v_new[0] * 255, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(v, v_new[1] * 255, 255, cv2.THRESH_BINARY_INV)
    background3 = cv2.bitwise_and(threshold, threshold2)
    background = cv2.bitwise_and(background1,background2,background3)
    if ktory == 5:
        background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((85, 85), np.uint8))
        background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((30, 30), np.uint8))
    elif ktory == 3:
        background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((10, 10), np.uint8))
        background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((70, 70), np.uint8))
    else:
        background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea)
    sort.reverse()
    mask = np.zeros(data.shape, np.uint8)
    i = 0
    hull_list = []
    for c in range(len(sort)):
        if cv2.contourArea(sort[c]) < 170000:
            i += 1
            hull = cv2.convexHull(sort[c])
            hull_list.append(hull)
            if i == ile:
                break
    cv2.drawContours(mask, hull_list, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(rawData, hull_list, -1, color, cv2.FILLED)
    if ktory == 6:
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((40, 40), np.uint8))
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((40, 40), np.uint8))
    mask = cv2.bitwise_not(mask)
    images = cv2.bitwise_and(data, mask)
    return images

def workOnImage(rawData):
    image = rawData.copy()
    contour = findBackground(image, [0.50, 0.65])
    image_size = image.shape[:2]
    mask = np.zeros(image_size, dtype=np.uint8)
    contour_hull = cv2.convexHull(contour, False)
    # If the contour is not solid draw the covex hull instead
    if cv2.contourArea(contour) > 0.5 * cv2.contourArea(contour_hull):
        mask = drawContourOnImage(mask, contour)
    else:
        mask = drawContourOnImage(mask, contour_hull)
    image = cutBackground(image, mask)
    image2 = image.copy()
    kernel = np.ones((30, 30), np.float32) / 900
    #image = cv2.bilateralFilter(image, 50, 250, 250)
    image = cv2.filter2D(image, -1, kernel)
    #image = cv2.blur(image, (40,40))
    #image = cv2.medianBlur(image, 15)

    image = findTerrain(rawData, image, (0, 255, 0), [0.15, 0.3], [0, 1], [0, 1], 4, 1)  # owce
    image = findTerrain(rawData, image, (0, 100, 0), [0.13, 0.2], [0.4, 1], [0, 0.4], 4, 2)  # las

    image_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_gray = a.apply(image_gray)
    thresh = [220, 255]
    _, threshold = cv2.threshold(image_gray, thresh[0], 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(image_gray, thresh[1], 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(threshold, threshold2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((10, 10), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((45, 45), np.uint8))
    mask = cv2.bitwise_not(mask)
    image = cutBackground(image, mask)

    resztkiWody = np.zeros(image_size, dtype=np.uint8)
    cont = findBackground(image, [0.4, 0.8])
    resztkiWody = drawContourOnImage(resztkiWody, cont)
    resztkiWody = cv2.bitwise_not(resztkiWody)
    image = cutBackground(image, resztkiWody)
    wynGory = findTerrain(rawData, image, (115, 115, 115), [0.07, 0.90], [0, 0.4], [0.3, 0.7], 3, 3)  # gory

    saturacja = cv2.cvtColor(wynGory, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(saturacja)
    s = s * 5
    s = np.clip(s, 0, 255)
    saturacja = cv2.merge([h, s, v])
    saturacja = cv2.cvtColor(saturacja.astype("uint8"), cv2.COLOR_HSV2BGR)

    image = findTerrain(rawData, saturacja, (0, 50, 185), [0.07, 0.1], [0, 10], [0.4, 0.95], 3, 4)  # glina

    tmp_images2.append(wynGory)
    image = cv2.cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((60, 60), np.uint8))
    image = cv2.cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((30, 30), np.uint8))
    image = findTerrain(rawData, image, (0, 135, 185), [0.1, 0.15], [0, 10], [0.75, 1], 1, 5) # pustynia
    tmp_images3.append(image)
    image = findTerrain(rawData, image, (0, 185, 255), [0, 0.25], [0, 10], [0.2, 0.75], 4, 6) #pola

    return rawData


def main():
    images = loadImages()
    for image in images:
        processed_images.append(image)
        data = image.copy()
        imageDone = workOnImage(data)
        tmp_images.append(imageDone)
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
            cv2.imshow('catan4', tmp_images3[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()