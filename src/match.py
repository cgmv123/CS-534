import numpy as np
import cv2

REPROJ_THRESH = 4
RATIO = .75


def get_matches(lkp, rkp, Lfeatures, Rfeatures):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(Lfeatures, Rfeatures, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * RATIO:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        ptsA = np.float32([lkp[i].pt for (_, i) in matches])
        ptsB = np.float32([rkp[i].pt for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, REPROJ_THRESH)
        return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None
