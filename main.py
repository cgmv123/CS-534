import cv2
import sys
from src import extract
from src import match
from src import stitch
import numpy as np
from src import input
from src import frames
from cv2 import createStitcher
import time

OUTPUT = "output/"
SCALE = .5
WARP_PARAM = 1650


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 4), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
            ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
            cv2.line(vis, ptA, ptB, (255, 0, 0), 3)

    # return the visualization
    return vis


def main():
    totTime = time.time()
    feature_method = sys.argv[2]

    kernel_size = [1, 2, 4, 8, 16, 32, 64]

    frms = frames.frames(sys.argv[1])
    # frms = frames.load_files(sys.argv[1])

    plotting = False

    imgs = []
    sTime = time.time()
    for img in frms:
        img = input.cylindricalWarp(img, WARP_PARAM)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGRA)
        imgs.append(img)
    eTime = time.time()
    print("Cylindrical Warping took " + str(eTime - sTime))

    num_images = len(imgs)

    # extract features
    sTime = time.time()
    features = []
    for img in imgs:
        features.append(extract.get_features(img, feature_method))
    eTime = time.time()
    print("Getting features took " + str(eTime - sTime))

    matches_list = []
    Hs = []
    tot_time = 0
    for i in range(0, num_images - 1):
        kpsA = features[i][0]
        # print("Num features: " + str(len(kpsA)))
        kpsB = features[i + 1][0]
        sTime = time.time()
        matches, H, status = match.get_matches(features[i][0], features[i + 1][0], features[i][1], features[i + 1][1])
        eTime = time.time()
        temp_time = eTime - sTime
        tot_time += temp_time
        matches_list.append((matches))
        Hs.append(np.linalg.inv(H))
        if plotting:
            img2 = cv2.drawKeypoints(cv2.cvtColor(imgs[i], cv2.COLOR_BGRA2BGR), kpsA, None)
            cv2.imwrite(OUTPUT + feature_method + "_kp_" + str(i) + ".png", img2)
            tmp = drawMatches(imgs[i], imgs[i + 1], kpsA, kpsB, matches, status)
            cv2.imwrite(OUTPUT + feature_method + "_matches_" + str(i) + ".png", tmp)
    print("Matching features took " + str(tot_time))
    i += 1
    kpsA = features[i][0]
    # print("Num features: " + str(len(kpsA)))
    if plotting:
        img2 = cv2.drawKeypoints(cv2.cvtColor(imgs[i], cv2.COLOR_BGRA2BGR), kpsA, None)
        cv2.imwrite(OUTPUT + feature_method + "_kp_" + str(i) + ".png", img2)

    # compound homographies
    H_map = []
    H_map.append(np.identity(3))
    for i in range(0, len(Hs)):
        H_map.append(np.matmul(H_map[i], Hs[i]))

    # stitch images
    sTime = time.time()
    result = stitch.stitch(imgs, H_map)
    eTime = time.time()
    print("Stiching took " + str(eTime - sTime))
    print(result.shape)

    cv2.imwrite(OUTPUT + feature_method + '_out.png', result)

    endTime = time.time()
    print("Total time: " + str(endTime - totTime))


if __name__ == "__main__":
    main()
