import cv2
import sys
from src import input
from src import extract
from src import match
from src import stitch
import time

def main(Limg_name, Rimg_name):

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
    Lfeatures = extract.get_features(Lgray)
    Rfeatures = extract.get_features(Rgray)
    eTime = time.time()

    print("Getting features took " + str(eTime - sTime))

    sTime = time.time()
    matches = match.get_matches(Lgray, Rgray, Lfeatures, Rfeatures)
    eTime = time.time()

    print("Getting matches took " + str(eTime - sTime))

    sTime = time.time()
    result = stitch.stitch(matches, Limg, Rimg, Lfeatures, Rfeatures)
    eTime = time.time()

    print("Stitching took " + str(eTime - sTime))

    cv2.imwrite('out.png', result)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])