import cv2
import numpy as np
import math

WARP_PARAM = 800


def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img


# # TODO CLEANUP
def cylindricalWarp(img):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""

    h_, w_ = img.shape[:2]
    K = np.array([[WARP_PARAM, 0, w_ / 2], [0, WARP_PARAM, h_ / 2], [0, 0, 1]])  # mock intrinsics
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    # B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[B[:, 1] < 0] == -1) | (B[:, 1] >= h_)] == -1
    B = B.reshape(h_, w_, -1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

