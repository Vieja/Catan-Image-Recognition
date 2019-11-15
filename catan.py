# from __future__ import division
from pylab import *
from skimage import data, io, filters, exposure, measure, feature
from skimage import img_as_float, img_as_ubyte
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb
from matplotlib import pylab as plt
import numpy as np
import os

io.use_plugin('matplotlib')


def loadImages():
    images = []
    main_dir = 'catan-photos'
    for directory in os.listdir(main_dir):
        for file_name in os.listdir(main_dir + '/' + directory):
            file_path = main_dir + '/' + directory + '/' + file_name
            images.append(imread(file_path))
    return images


def workOnImage(image):
    imgSize = image.shape[:2]
    data = img_as_float(image, imgSize)
    # ///////////////

    contour = removeBackground(data, imgSize)
    showImageWithContour(image, contour)


def showImage(img):
    io.imshow(img)
    plt.axis('off')
    plt.show()


def showImageWithContour(img, contour):
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    ax.step(contour.T[1], contour.T[0], linewidth=2, c='red')
    plt.show()


# https://plot.ly/python/v3/polygon-area/
def polygonArea(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def removeBackground(data, imgSize):
    min = np.percentile(data, 5)
    max = np.percentile(data, 95)
    data = exposure.rescale_intensity(data, in_range=(min, max))

    # showImage(data)
    dataHSV = rgb2hsv(data)
    blue = [0.50, 0.65]  # zakres wartosci H dla niebieskiego w HSV
    maska = np.zeros(imgSize, dtype=np.float)
    for row, line in enumerate(dataHSV):
        for col, pixel in enumerate(line):
            if blue[0] < pixel[0] < blue[1]:
                maska[row][col] = 1

    maska = mp.dilation(maska)
    showImage(maska)
    contours = measure.find_contours(maska, 0.3)

    # contour = sorted(contours, key=lambda x: polygonArea(x))[-1]
    biggestContourIndex = 0
    biggestContourSize = 0
    for i, cont in enumerate(contours):
        currentContourSize = polygonArea(cont)
        if currentContourSize > biggestContourSize:
            biggestContourIndex = i
            biggestContourSize = currentContourSize
    return contours[biggestContourIndex]


images = loadImages()
for img in images:
    workOnImage(img)
