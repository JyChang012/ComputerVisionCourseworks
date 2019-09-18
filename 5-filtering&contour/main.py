import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import cv2 as cv


def bilateral_filter(src=np.array([]), kernel_size=(3, 3), sigma_color=1., sigma_space=1.):

    src = src.astype(np.float64)

    # Construct space kernel
    space_kernel = np.zeros(kernel_size)
    kernel_size = np.array(kernel_size)
    center = (kernel_size - 1) / 2
    for i in range(kernel_size[0]):
        for j in range(kernel_size[1]):
            space_kernel[i, j] = np.linalg.norm(np.array([i, j]) - center)
    space_kernel = scipy.stats.norm.pdf(space_kernel, loc=0, scale=sigma_space)
    space_kernel = space_kernel / space_kernel.sum()

    if len(src.shape) > 2:
        space_kernel = space_kernel[..., np.newaxis]

    border_vertical = int((kernel_size[0] - 1) / 2)
    border_horizontal = int((kernel_size[1] - 1) / 2)

    src_with_border = cv.copyMakeBorder(src, border_vertical, border_vertical, border_horizontal, border_horizontal,
                                        cv.BORDER_CONSTANT)

    dst = np.zeros(src.shape, dtype=src.dtype)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            roi = src_with_border[i:i + kernel_size[0], j:j + kernel_size[1]]

            color_kernel = roi - src[i, j]

            if len(src.shape) > 2:
                color_kernel = np.linalg.norm(color_kernel, axis=2, keepdims=True)
            else:
                color_kernel = np.abs(color_kernel)

            color_kernel = scipy.stats.norm.pdf(color_kernel, loc=0, scale=sigma_color)
            color_kernel = color_kernel / color_kernel.sum()

            kernel = space_kernel * color_kernel
            kernel = kernel / kernel.sum()

            # dst[i, j] = np.average(roi, axis=(0, 1), weights=kernel)
            dst[i, j] = np.sum(roi * kernel, axis=(0, 1))

    return dst.astype(np.uint8)


def test():
    img = cv.imread('selfie.jpg')
    dst = bilateral_filter(img, kernel_size=[7, 7], sigma_color=.25, sigma_space=8)
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()


def test_gray():
    img = cv.imread('lena.jpg', 0)
    dst = bilateral_filter(img, kernel_size=[7, 7], sigma_color=.25, sigma_space=18)
    plt.imshow(dst, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


def test_flower_api():
    img = cv.imread('flower.jpg')
    dst = cv.bilateralFilter(img, 9, 75, 75)
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()


def test_flower():
    img = cv.imread('flower.jpg')
    dst = bilateral_filter(img, (9, 9), 75, 75)
    plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    test_flower()
