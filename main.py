import cv2
import sys
from src import input
from src import extract
from src import match
from src import stitch
import time
import numpy as np

def main(Limg_name, Rimg_name, feature_method):

    sTime = time.time()
    Limg = cv2.imread(Limg_name)
    Rimg = cv2.imread(Rimg_name)
    eTime = time.time()

    print("Reading took " + str(eTime - sTime))

    sTime = time.time()
    Limg = input.equalize_histogram_color(Limg)
    Rimg = input.equalize_histogram_color(Rimg)

    Lgray = cv2.cvtColor(Limg,cv2.COLOR_BGR2GRAY)
    Rgray = cv2.cvtColor(Rimg,cv2.COLOR_BGR2GRAY)
    eTime = time.time()

    print("Preprocessing took " + str(eTime - sTime))

    sTime = time.time()
    Lfeatures = extract.get_features(Lgray, feature_method)
    Rfeatures = extract.get_features(Rgray, feature_method)
    eTime = time.time()

    print("Getting features took " + str(eTime - sTime))

    sTime = time.time()
    matches, H, status, ptsA, ptsB = match.get_matches(Lgray, Rgray, Lfeatures, Rfeatures, feature_method)
    eTime = time.time()

    print("Getting matches took " + str(eTime - sTime))

    # for pt in Lfeatures[0]:
    #     cv2.circle(Limg, (int(pt.pt[0]),int(pt.pt[1])), 3, (0,0,255), -1)
    #
    # for pt in ptsA:
    #     cv2.circle(Limg, (pt[0], pt[1]), 3, 255, -1)
    #
    # for pt in Rfeatures[0]:
    #     cv2.circle(Rimg, (int(pt.pt[0]), int(pt.pt[1])), 3, (0,0,255), -1)
    #
    # for pt in ptsB:
    #     cv2.circle(Rimg, (pt[0], pt[1]), 3, 255, -1)
    #
    # result = np.concatenate((Limg, Rimg), axis=1)
    #
    # cv2.imshow("Limg", result)
    # cv2.waitKey(0)



    sTime = time.time()
    result = stitch.stitch(Rimg, Limg, H)
    eTime = time.time()

    print("Stitching took " + str(eTime - sTime))

    cv2.imwrite('out.png', result)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "sift")