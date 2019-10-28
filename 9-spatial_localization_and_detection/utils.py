import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

bounding2xywh = np.array([[.5, 0, .5, 0],
                          [0, .5, 0, .5],
                          [-1, 0, 1, 0],
                          [0, -1, 0, 1]])


def plot_box_from_xywh(xywh):
    bounding = bounding_xywh_transform(xywh, b2xywh=False)
    plot_box_from_min_max(bounding)


def plot_box_from_min_max(bounding):  # bounding = [xmin, ymin, xmax, ymax]
    x = [bounding[0], bounding[0], bounding[2], bounding[2], bounding[0]]
    y = [bounding[1], bounding[3], bounding[3], bounding[1], bounding[1]]
    plt.plot(x, y)


def bounding_xywh_transform(coordinate, b2xywh=True, batch=False):
    """Transform between (x, y, w, h) and (xmin, ymin, xmax, ymax) representations of bounding box."""
    if not batch:
        if b2xywh:
            return bounding2xywh @ coordinate
        else:
            return np.linalg.inv(bounding2xywh) @ coordinate
    else:
        if b2xywh:
            return np.vstack([bounding2xywh @ x for x in coordinate])
        else:
            mat = tf.linalg.inv(bounding2xywh)
            return np.vstack([mat @ x for x in coordinate])


if __name__ == '__main__':
    img = plt.imread('./tiny_vid/bird/000001.JPEG')
    bird_annote = np.loadtxt('./tiny_vid/bird_gt.txt', dtype=np.int)
    # each line is (image_index, xmin, ymin, xmax, ymax), where x is horizontal coordinate and y is vertical with origin
    # at upper left
    plt.imshow(img)
    plot_box_from_xywh(bounding_xywh_transform(bird_annote[0, 1:]))
    plt.show()
