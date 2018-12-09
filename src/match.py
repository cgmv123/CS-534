import numpy as np
import time
import cv2
import scipy
from pprint import pprint

DSIZE = 8

YTHRESH = 100
RTHRESH = .05
CORRTHRESH = .95

REPROJ_THRESH = 4

def get_descriptor(img, x, y):
    y0 = int(x - DSIZE/2)
    x0 = int(y - DSIZE/2)
    window = img[x0:x0+DSIZE, y0:y0+DSIZE]
    return window

def calc_corr(xl, yl, xr, yr, Limg, Rimg):
    Lwindow = get_descriptor(Limg, xl, yl)
    Rwindow = get_descriptor(Rimg, xr, yr)
    if Lwindow.shape != (DSIZE,DSIZE) or Rwindow.shape != (DSIZE,DSIZE):
        return 0
    Lmean = np.mean(Lwindow)
    Rmean = np.mean(Rwindow)
    top = 0
    for u in range(0,DSIZE):
        for v in range(0,DSIZE):
            top += (Lwindow[u,v] - Lmean) * (Rwindow[u,v] - Rmean)

    Lbottom = 0
    Rbottom = 0
    for u in range(0,DSIZE):
        for v in range(0,DSIZE):
            Lbottom += (Lwindow[u,v] - Lmean)**2
            Rbottom += (Rwindow[u,v] - Rmean)**2

    return top / (np.sqrt(Lbottom * Rbottom))

def nearest_neighbors_kd_tree(x, y, k):
    x, y = map(np.asarray, (x, y))
    tree = scipy.spatial.cKDTree(y[:, None])
    ordered_neighbors = tree.query(x[:, None], k)[1]
    nearest_neighbor = np.empty((len(x),), dtype=np.intp)
    nearest_neighbor.fill(-1)
    used_y = set()
    for j, neigh_j in enumerate(ordered_neighbors):
        for k in neigh_j:
            if k not in used_y:
                nearest_neighbor[j] = k
                used_y.add(k)
                break
    return nearest_neighbor

def corner_sim(Limg, Rimg, Lfeatures, Rfeatures):
    l_len = len(Lfeatures)
    r_len = len(Rfeatures)

    C_mat = np.zeros((l_len, r_len))


    i = 0
    stime = time.time()
    count = 0
    for Lfeature in Lfeatures:
        xl = Lfeature[0]
        yl = Lfeature[1]
        Rl = Lfeature[2]
        j = 0
        for Rfeature in Rfeatures:
            xr = Rfeature[0]
            yr = Rfeature[1]
            Rr = Rfeature[2]

            # NEED BETTER CONSTRAINTS
            if xl >= xr and np.abs(yr -yl) < YTHRESH and np.abs((Rr - Rl)/Rl) < RTHRESH:
                count += 1
                corr = calc_corr(xl, yl, xr, yr, Limg, Rimg)
                if (np.abs(corr) > CORRTHRESH):
                    C_mat[i,j] = np.abs(calc_corr(xl, yl, xr, yr, Limg, Rimg))
            j+=1
        i+=1
    etime = time.time()
    print(count)
    print("CORR MAT took " +  str(etime - stime))
    return C_mat


def get_matches(Limg, Rimg, Lfeatures, Rfeatures, feature_method):

    if feature_method == "custom":
        return get_matches_custom(Limg, Rimg, Lfeatures, Rfeatures)
    elif feature_method == "sift":
        return get_matches_SIFT(Lfeatures[0], Rfeatures[0], Lfeatures[1], Rfeatures[1])



def get_matches_custom(Limg, Rimg, Lfeatures, Rfeatures):
    C_mat = corner_sim(Limg, Rimg, Lfeatures, Rfeatures)
    C_matt = np.transpose(C_mat.copy())

    dims = C_mat.shape

    matchesL = np.zeros(dims[0])
    i = 0
    for row in C_mat:
        matchesL[i] = np.argmax(row)
        i += 1

    matchesR = np.zeros(dims[1])
    i = 0
    for row in C_matt:
        matchesR[i] = np.argmax(row)
        i += 1

    for i in range(0, len(matchesL)):
        matchL = matchesL[i]
        matchR = matchesR[int(matchL)]

        if matchR != i:
            matchesL[i] = 0
            matchesR[int(matchL)] = 0

    return get_homography(matchesL, Lfeatures, Rfeatures)

def get_matches_SIFT(lkp, rkp, Lfeatures, Rfeatures):
    # compute the raw matches and initialize the list of actual
    # matches
    ratio = .75
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(Lfeatures, Rfeatures, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([lkp[i].pt for (_, i) in matches])
        ptsB = np.float32([rkp[i].pt for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, REPROJ_THRESH)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status, ptsA, ptsB)

    # otherwise, no homograpy could be computed
    return None



def get_homography(matches, Lfeatures, Rfeatures):
    ptsL = []
    ptsR = []

    i = 0
    for match in matches:
        if match != 0:
            xl = Lfeatures[i][0]
            yl = Lfeatures[i][1]
            ptsL.append((xl, yl))
            xr = Rfeatures[int(match)][0]
            yr = Rfeatures[int(match)][1]
            ptsR.append((xr, yr))
        i += 1

    (H, status) = cv2.findHomography(np.asarray(ptsL), np.asarray(ptsR), cv2.RANSAC, REPROJ_THRESH)

    return (matches, H, status)