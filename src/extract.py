import cv2
import numpy as np
import bottleneck as bn
import src.match as match

# CUSTOM
NUM_HEIGHT = 5
NUM_WIDTH = 5

TAU = 10
NUM_CORNERS = 5000

# HARRIS

# GFT

# SIFT
N_FEATURES = 5000
N_OCTAVE_LAYERS_SIFT = 3
CONTRAST_THRESHOLD = 0.001
EDGE_THRESHOLD = 5
SIGMA = 1.6

# SURF
HESSIAN_THRESHOLD = 200
N_OCTAVES = 5
N_OCTAVE_LAYERS_SURF = 3
EXTENDED = True
UPRIGHT = False


def sigmoid(var, max, min, mean):
    return 1 / (1 + np.exp(-TAU * (var - mean) / (max - min)))


def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def get_features(img, feature_method):
    if feature_method == "custom":
        return get_features_custom(img)
    elif feature_method == "sift":
        return get_features_SIFT(img)
    elif feature_method == "surf":
        return get_features_SURF(img)
    elif feature_method == "harris":
        return get_features_harris(img)
    elif feature_method == "gfft":
        return get_features_GFTT(img)


def get_features_custom(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    box_height = (height / NUM_HEIGHT)
    box_width = (width / NUM_WIDTH)

    vars = np.zeros((NUM_HEIGHT, NUM_WIDTH))
    sigs = np.zeros((NUM_HEIGHT, NUM_WIDTH))
    weights = np.zeros((NUM_HEIGHT, NUM_WIDTH))

    i = 0
    for y in np.arange(0, height, box_height):
        j = 0
        for x in np.arange(0, width, box_width):
            vars[i, j] = np.var(img[int(y):int(y + box_height), int(x):int(x + box_width)])
            j += 1
        i += 1

    maxvar = np.amax(vars)
    minvar = np.amin(vars)
    meanvar = np.mean(vars)

    for i in np.arange(0, NUM_HEIGHT):
        for j in np.arange(0, NUM_WIDTH):
            sigs[i, j] = sigmoid(vars[i, j], maxvar, minvar, meanvar)

    sum_sigs = np.sum(sigs)

    for i in np.arange(0, NUM_HEIGHT):
        for j in np.arange(0, NUM_WIDTH):
            weights[i, j] = sigs[i, j] / sum_sigs

    i = 0
    keypoints = []
    for y in np.arange(0, height, box_height):
        j = 0
        for x in np.arange(0, width, box_width):
            num = int(NUM_CORNERS * weights[i, j])
            corners = cv2.cornerHarris(img[int(y):int(y + box_height), int(x):int(x + box_width)], 2, 3, 0.04)
            if num == 0:
                break
            idxs = top_n_indexes(corners, num)
            for idx in idxs:
                keypoints.append(cv2.KeyPoint(x + idx[1], y + idx[0], 1))
            j += 1
        i += 1
    sift = cv2.xfeatures2d.SIFT_create(N_FEATURES, N_OCTAVE_LAYERS_SIFT, CONTRAST_THRESHOLD, EDGE_THRESHOLD, SIGMA)
    features = sift.compute(img, keypoints)

    return features


def get_features_harris(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    corners = cv2.cornerHarris(img, 2, 3, 0.04)
    sift = cv2.xfeatures2d.SIFT_create(N_FEATURES, N_OCTAVE_LAYERS_SIFT, CONTRAST_THRESHOLD, EDGE_THRESHOLD, SIGMA)
    keypoints = []
    idxs = top_n_indexes(corners, NUM_CORNERS)
    for idx in idxs:
        keypoints.append(cv2.KeyPoint(idx[1], idx[0], 1))
    features = sift.compute(img, keypoints)
    return features


def get_features_GFTT(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    corners = cv2.goodFeaturesToTrack(img, NUM_CORNERS, .001, 2)
    sift = cv2.xfeatures2d.SIFT_create(N_FEATURES, N_OCTAVE_LAYERS_SIFT, CONTRAST_THRESHOLD, EDGE_THRESHOLD, SIGMA)
    keypoints = []
    for idx in corners:
        x, y = idx.ravel()
        keypoints.append(cv2.KeyPoint(x, y, 1))
    features = sift.compute(img, keypoints)
    return features


def get_features_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create(N_FEATURES, N_OCTAVE_LAYERS_SIFT, CONTRAST_THRESHOLD, EDGE_THRESHOLD, SIGMA)
    (kps, features) = sift.detectAndCompute(img, None)
    return [kps, features]


def get_features_SURF(img):
    surf = cv2.xfeatures2d.SURF_create(HESSIAN_THRESHOLD, N_OCTAVES, N_OCTAVE_LAYERS_SURF, EXTENDED, UPRIGHT)
    (kps, features) = surf.detectAndCompute(img, None)
    return [kps, features]
