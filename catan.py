from __future__ import division
from pylab import *
from skimage import data, io, filters, exposure, measure, feature
from skimage import img_as_float, img_as_ubyte
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb
from matplotlib import pylab as plt
import numpy as np

io.use_plugin('matplotlib')


def loadImages():
    images = []
    for i in range(1, 9):
        filename = "catan-photos/2players/" + str(i) + ".jpg"
        images.append(io.imread(filename))
        i += 1
    i = 1
    for i in range(1, 11):
        filename = "catan-photos/3+players/" + str(i) + ".jpg"
        images.append(io.imread(filename))
        i += 1

    # test = ["pola", "liczby"]
    # for filename in test:
    #     filename = "catan-photos/test/" + filename + ".jpg"
    #     images.append(io.imread(filename))
    return images


def workOnImage(image):
    data = img_as_float(image)
    # ///////////////

    contour = removeBackground(data)
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


def removeBackground(data):
    min = np.percentile(data, 5)
    max = np.percentile(data, 95)
    data = exposure.rescale_intensity(data, in_range=(min, max))

    #showImage(data)
    dataHSV = rgb2hsv(data)
    blue = [0.50, 0.65]  # zakres wartosci H dla niebieskiego w HSV
    tempMatrix = []
    for line in dataHSV:
        temp = []
        for pixel in line:
            if pixel[0] > blue[0] and pixel[0] < blue[1]:
                temp.append(1)
            else:
                temp.append(0)
        tempMatrix.append(temp)
    maska = np.array(tempMatrix)
    maska = mp.dilation(maska)
    showImage(maska)
    contours = measure.find_contours(maska, 0.3)
    contour = sorted(contours, key=lambda x: polygonArea(x))[-1]
    return contour


images = loadImages()
for img in images:
    workOnImage(img)