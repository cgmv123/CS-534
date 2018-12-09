import cv2
import sys
from src import extract
from src import match
from src import stitch
import numpy as np


def main():
    feature_method = "sift"

    # read in images
    imgs = []
    gray_imgs = []
    first_arg = True
    for img_name in sys.argv:
        if first_arg:
            first_arg = False
        else:
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img)
            gray_imgs.append(gray_img)

    num_images = len(imgs)

    # extract features
    features = []
    for img in imgs:
        features.append(extract.get_features(img, feature_method))

    # match features
    matches_list = []
    Hs = []
    ptsL_list = []
    ptsR_list = []
    for i in range(0, num_images - 1):
        Lgray = gray_imgs[i]
        Rgray = gray_imgs[i + 1]
        matches, H, status, ptsL, ptsR = match.get_matches(Lgray, Rgray, features[i], features[i + 1], feature_method)
        matches_list.append((matches))
        Hs.append(H)
        ptsL_list.append(ptsL)
        ptsR_list.append(ptsR)

    # compound homographies
    H_map = []
    H_map.append(np.identity(3))
    for i in range(0, len(Hs)):
        H_map.append(np.matmul(H_map[i], Hs[i]))

    # stitch images
    result = stitch.stitch(imgs, H_map)

    cv2.imwrite('out.png', result)


if __name__ == "__main__":
    main()
