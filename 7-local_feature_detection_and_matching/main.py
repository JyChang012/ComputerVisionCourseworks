import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def corner(filename, save=False):
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, .04)
    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if save:
        plt.imsave(save, cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    # plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))


if __name__ == '__main__':
    for file in ['IMG_4495.JPG', 'IMG_4494.JPG', 'IMG_4496.JPG']:
        corner(file, 'output'+file)
