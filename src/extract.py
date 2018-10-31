import cv2
import numpy as np
import bottleneck as bn
from src import match

NUM_HEIGHT = 10
NUM_WIDTH = 10

TAU = 10
NUM_CORNERS = 1000

DSIZE = 8

YTHRESH = 200
RTHRESH = .05
CORRTHRESH = .9

LENGTH_THRESH = 10
SLOPE_THRESH = 5

def sigmoid(var, max, min, mean):
    return 1 / (1 + np.exp(-TAU*(var-mean)/(max - min)))


def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def get_features(img):
    height = img.shape[0]
    width = img.shape[1]
    box_height = int(height / NUM_HEIGHT)
    box_width = int(width / NUM_WIDTH)

    vars = np.zeros((NUM_HEIGHT,NUM_WIDTH))
    sigs = np.zeros((NUM_HEIGHT,NUM_WIDTH))
    weights = np.zeros((NUM_HEIGHT,NUM_WIDTH))
    features = []

    i = 0
    for y in np.arange(0,height, box_height):
        j = 0
        for x in np.arange(0, width, box_width):
            vars[i,j] = np.var(img[y:y+box_height, x:x+box_width])
            j += 1
        i += 1

    maxvar = np.amax(vars)
    minvar = np.amin(vars)
    meanvar = np.mean(vars)

    for i in np.arange(0,NUM_HEIGHT):
        for j in np.arange(0, NUM_WIDTH):
            sigs[i,j] = sigmoid(vars[i,j], maxvar, minvar, meanvar)

    sum_sigs = np.sum(sigs)

    for i in np.arange(0,NUM_HEIGHT):
        for j in np.arange(0, NUM_WIDTH):
            weights[i,j] = sigs[i,j] / sum_sigs

    i = 0
    for y in np.arange(0,height, box_height):
        j = 0
        for x in np.arange(0, width, box_width):
            num = int(NUM_CORNERS*weights[i,j])
            corners = cv2.cornerHarris(img[y:y+box_height,x:x+box_width],2,31,0.04)
            if num == 0:
                break
            idxs = top_n_indexes(corners, num)
            for idx in idxs:
                features.append([idx[1]+x,idx[0]+y, corners[idx]])
            j+=1
        i+=1

    return features