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


def thresholdBetweenValues(image, thresh_min, thresh_max):
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(image, thresh_min, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(image, thresh_max, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(threshold, threshold2)


def thresholdInRange(image, threshold_range):
    return thresholdBetweenValues(image, threshold_range[0], threshold_range[1])


def drawContourOnImage(image, contour):
    cv2.drawContours(image, [contour], -1, 255, cv2.FILLED)
    return image


def findBackground(image):
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    blue = [0.50 * 180, 0.65 * 180]
    background = thresholdInRange(h, blue)
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


def findFields(image, red_mask, blue_mask):
    hex_shape = np.load('hex.npy')
    # a = findCircles(image)
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_gray = np.uint8(cv2.pow(image_gray/255, 1/2)*255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
    thresh = [170, 255]
    mask = thresholdInRange(image_gray, thresh)
    # tmp_images2.append(mask)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((10, 10), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((15, 15), np.uint8))
    # tmp_images3.append(mask)
    # mask = cv2.add(mask, pieces_mask)
    mask = cv2.add(mask, red_mask)
    mask = cv2.add(mask, blue_mask)
    # mask = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # sort = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [x for x in contours if 30000 < cv2.contourArea(x) < 500000 and
                cv2.matchShapes(cv2.convexHull(x, False), hex_shape, 1, 0.0) < 0.03]
    # found_fields_num = 0
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Usuwanie znalezionych pól, żeby kolejna metoda szukania nie znalazła znowu tych samych
    found_fields_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    for contour in contours:
        # if cv2.contourArea(contours[c]) > 1:
        # contour_hull = cv2.convexHull(contours[c], False)
        # if c < 20:
        # print(cv2.contourArea(sort[c]))
        random_color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        hull = cv2.convexHull(contour, False)
        # cv2.drawContours(image, sort, c, random_color, cv2.FILLED)
        cv2.drawContours(image, [hull], -1, random_color, cv2.FILLED)
        cv2.drawContours(found_fields_mask, [hull], -1, 0, cv2.FILLED)
        # found_fields_num += 1
    image = cv2.bitwise_and(image, image, mask=found_fields_mask)
    found_fields_num = len(contours)
    print("Found {}/19 fields".format(found_fields_num))
    if found_fields_num == 19:
        return image

    # Seconds pass
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    thresh = [100, 120]
    image_gray = cv2.equalizeHist(s)
    mask = thresholdInRange(image_gray, thresh)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Take only contours that are ok
    contours = [x for x in contours if
                cv2.contourArea(x) > 70000 and (cv2.matchShapes(x, hex_shape, 1, 0.0) < 0.03)]
    for c in contours:
        convex_hull = cv2.convexHull(c, False)
        cv2.drawContours(image, [convex_hull], -1, 0, cv2.FILLED)
        cv2.drawContours(found_fields_mask, [convex_hull], -1, 0, cv2.FILLED)
    found_fields_num += len(contours)
    image = cv2.bitwise_and(image, image, mask=found_fields_mask)
    print("Found {}/19 fields. It's {} more than before.".format(found_fields_num, len(contours)))
    if found_fields_num == 19:
        return image
    return image


def findCircles(data):
    h, s, v = cv2.split(cv2.cvtColor(data, cv2.COLOR_BGR2HSV))
    h_new = [0.1 * 180, 0.12 * 180]
    s_new = [0.4 * 255, 0.5 * 255]
    v_new = [0.7 * 255, 1 * 255]
    # if (ktory == 3):
    # data[:, 2, :] = 255
    background1 = thresholdInRange(h, h_new)
    background2 = thresholdInRange(s, s_new)
    background3 = thresholdInRange(v, v_new)
    background = cv2.bitwise_and(background1, background2, background3)

    background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    # background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea)
    sort.reverse()
    mask = np.zeros(data.shape, np.uint8)
    i = 0
    hull_list = []
    for c in range(len(sort)):
        i += 1
        hull = cv2.convexHull(sort[c])
        hull_list.append(hull)
    cv2.drawContours(mask, hull_list, -1, (255, 255, 255), cv2.FILLED)
    # cv2.drawContours(rawData, hull_list, -1, color, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((20, 20), np.uint8))
    mask = cv2.bitwise_not(mask)
    images = cv2.bitwise_and(data, mask)
    tmp_images2.append(background)
    # tmp_images3.append(mask)
    return images


# def removePieces(image):
#     # Jak na razie tylko 'proof of concept' działa jedynie dla niebieskich pionków i to też nie do końca
#     image_size = image.shape[:2]
#     blue_mask = np.zeros(image_size, dtype=np.uint8)
#     blue_range = [0, 2]
#     l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
#     image_gray = cv2.equalizeHist(b)
#     _, threshold = cv2.threshold(image_gray, blue_range[0], 255, cv2.THRESH_BINARY)
#     _, threshold2 = cv2.threshold(image_gray, blue_range[1], 255, cv2.THRESH_BINARY_INV)
#     tmp = cv2.bitwise_and(threshold, threshold2)
#     contours, hierarchy = cv2.findContours(tmp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     for i, cont in enumerate(contours):
#         if cv2.contourArea(cont) > 100:
#             hull = cv2.convexHull(cont)
#             blue_mask = cv2.drawContours(blue_mask, [hull], -1, 255, cv2.FILLED)
#     mask_all = blue_mask  # Join all masks together
#     # tmp_images3.append(mask_all)
#     return mask_all

def findRedPieces(data):
    h, s, v = cv2.split(cv2.cvtColor(data, cv2.COLOR_BGR2HSV))
    # if (ktory == 3):
    # data[:, 2, :] = 255
    # Finding two thresholds and then finding the common part
    h_new = [0.07 * 180, 0.93 * 180]
    s_new = [0.5 * 255, 1 * 255]
    v_new = [0 * 255, 1 * 255]

    _, threshold = cv2.threshold(h, h_new[0], 180, cv2.THRESH_BINARY_INV)
    _, threshold2 = cv2.threshold(h, h_new[1], 180, cv2.THRESH_BINARY)
    background1 = cv2.bitwise_xor(threshold, threshold2)
    background2 = thresholdInRange(s, s_new)
    background3 = thresholdInRange(v, v_new)
    background = cv2.bitwise_and(background1, background2, background3)
    return background


def removeRedPieces(data):
    background = findRedPieces(data)
    # tmp_images2.append(background)
    # jeżeli chcemy tym wykrywać piony to trzeba to zrobic tu, przed dylacją
    background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((20, 20), np.uint8))
    background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((70, 70), np.uint8))
    # tmp_images3.append(background)
    # background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea)
    sort.reverse()
    mask = np.zeros(data.shape[:2], np.uint8)
    hull_list = []
    for c in range(len(sort)):
        if cv2.contourArea(sort[c]) > 1000:
            hull = cv2.convexHull(sort[c])
            hull_list.append(hull)
    # cv2.drawContours(mask, hull_list, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    # cv2.drawContours(rawData, hull_list, -1, color, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((20, 20), np.uint8))
    # images = cv2.bitwise_and(data, mask)
    #     tmp_images3.append(mask)
    return mask


def findBluePieces(data):
    h, s, v = cv2.split(cv2.cvtColor(data, cv2.COLOR_BGR2HSV))
    # if (ktory == 3):
    # data[:, 2, :] = 255
    # Finding two thresholds and then finding the common part
    h_new = [0.58 * 180, 0.69 * 180]
    s_new = [0.3 * 255, 1 * 255]
    v_new = [0 * 255, 0.6 * 255]
    background1 = thresholdInRange(h, h_new)
    background2 = thresholdInRange(s, s_new)
    background3 = thresholdInRange(v, v_new)
    background = cv2.bitwise_and(background1, background2, background3)
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # tmp_images2.append(background)
    return background


def removeBluePieces(data):
    background = findBluePieces(data)
    # background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((20, 20), np.uint8))
    background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((30, 30), np.uint8))
    # tmp_images3.append(background)
    # background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea)
    sort.reverse()
    mask = np.zeros(data.shape[:2], np.uint8)
    hull_list = []
    for c in range(len(sort)):
        if cv2.contourArea(sort[c]) > 1000:
            hull = cv2.convexHull(sort[c])
            hull_list.append(hull)
    # cv2.drawContours(mask, hull_list, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)
    # cv2.drawContours(rawData, hull_list, -1, color, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((20, 20), np.uint8))
    # images = cv2.bitwise_and(data, mask)
    #     tmp_images3.append(mask)
    return mask


def identifyPieces(image, pieces_mask, piece_color):
    if piece_color == 'red':
        pieces_colors = [(0, 0, 255), (255, 0, 255)]
    else:
        pieces_colors = [(255, 0, 0), (255, 255, 0)]
    contours, hierarchy = cv2.findContours(pieces_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        if 500 < hull_area < 15000:
            # red_mask = cv2.drawContours(red_mask, [hull], -1, 255, cv2.FILLED)
            try:
                ellipse = cv2.fitEllipse(hull)
            except cv2.error:  # Za mały kontur aby wpasować elipsę
                continue
            (x, y), (Ma, ma), angle = ellipse
            if Ma / ma > 0.5:  # Jeśli osie elipsy są prawie równe mamy okrąg
                color = pieces_colors[0]
            else:  # Obiekt jest podłużny
                color = pieces_colors[1]
            cv2.drawContours(image, [hull], -1, color, cv2.FILLED)
        elif 15000 < hull_area < 50000:  # Jeśli kontur jest za duży, to być może dwa pionki się złączyły
            new_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            new_mask = cv2.drawContours(new_mask, [cont], -1, 255, cv2.FILLED)
            new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
            image = identifyPieces(image, new_mask, piece_color)
    return image


# def removeFrame(image):
#     h, w = image.shape[:2]
#     if h > w:
#         big = h
#     else:
#         big = w
#     dim = (w, h)
#     cropp = int (0.2 * big)
#     cropped = image[cropp : h-cropp, cropp : w-cropp]
#     # perform the actual resizing of the image and show it
#     resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
#     #tmp_images2.append(resized)
#     return image

def findTerrain(rawData, data, color, h_new, s_new, v_new, ile, ktory):
    h, s, v = cv2.split(cv2.cvtColor(data, cv2.COLOR_BGR2HSV))
    # if (ktory == 3):
    # data[:, 2, :] = 255
    background1 = thresholdBetweenValues(h, h_new[0] * 180, h_new[1] * 180)
    background2 = thresholdBetweenValues(s, s_new[0] * 255, s_new[1] * 255)
    background3 = thresholdBetweenValues(v, v_new[0] * 255, v_new[1] * 255)
    background = cv2.bitwise_and(background1, background2, background3)

    background = cv2.morphologyEx(background, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))
    # background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((2, 2), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sort = sorted(contours, key=cv2.contourArea)
    sort.reverse()
    mask = np.zeros(data.shape, np.uint8)
    i = 0
    hull_list = []
    for c in range(len(sort)):
        i += 1
        hull = cv2.convexHull(sort[c])
        hull_list.append(hull)
        if i == ile:
            break
    cv2.drawContours(mask, hull_list, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(rawData, hull_list, -1, color, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((20, 20), np.uint8))
    mask = cv2.bitwise_not(mask)
    images = cv2.bitwise_and(data, mask)
    # if ktory == 3:
    #     tmp_images2.append(background)
    #     tmp_images3.append(mask)
    return images


def workOnImage(rawData):
    image = rawData.copy()
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
    kernel = np.ones((30, 30), np.float32) / 900
    # image = cv2.bilateralFilter(image, 50, 250, 250)
    # image = cv2.filter2D(image, -1, kernel)
    # image = cv2.blur(image, (40,40))
    # image = cv2.medianBlur(image, 15)

    # image = findTerrain(rawData, image, (0, 255, 0), [0.15, 0.3], [0, 1], [0, 1], 4, 1)  # owce
    # image = findTerrain(rawData, image, (0, 100, 0), [0.13, 0.2], [0.4, 1], [0, 0.4], 4, 2)  # las
    # image = findTerrain(rawData, image, (115, 115, 115), [0.0, 1], [0, 0.2], [0.4, 0.7], 3, 3)  # gory
    # image = findTerrain(rawData, image, (115, 115, 115), [0.15, 0.28], [0.42, 0.7], [0, 1], 3, 3)  # gory
    blue_pieces = findBluePieces(image)
    red_pieces = findRedPieces(image)
    blue_mask = removeBluePieces(image)
    red_mask = removeRedPieces(image)
    rawData = findFields(image, red_mask, blue_mask)
    # Z jednej strony jeśli tutaj ponownie będziemy szukać pionków, to unikniemy mylenia gliny z czerwonym pionkiem
    # Z drugiej, jeśli któreś pole ma zbyt duży kontur i wycięło pionek, to stracimy ten pionek
    # blue_pieces = findBluePieces(image)
    # red_pieces = findRedPieces(image)
    rawData = identifyPieces(rawData, blue_pieces, 'blue')
    rawData = identifyPieces(rawData, red_pieces, 'red')
    return rawData


def main():
    images = loadImages()
    for i, image in enumerate(images):
        print("Processing image {}/{}".format(i + 1, len(images)))
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

