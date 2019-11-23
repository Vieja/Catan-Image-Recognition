import os
import cv2
import numpy as np
import random as rng


def main():
    hex_image = cv2.imread('hex.png')
    hex_image = cv2.cvtColor(hex_image, cv2.COLOR_BGR2GRAY)
    cont, _ = cv2.findContours(hex_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    np.save('hex', cont[0])

if __name__ == "__main__":
    main()