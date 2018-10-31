import numpy as np

DSIZE = 8

YTHRESH = 100
RTHRESH = .05
CORRTHRESH = .95

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


def corner_sim(Limg, Rimg, Lfeatures, Rfeatures):
    l_len = len(Lfeatures)
    r_len = len(Rfeatures)

    C_mat = np.zeros((l_len, r_len))
    i = 0
    for Lfeature in Lfeatures:
        xl = Lfeature[0]
        yl = Lfeature[1]
        Rl = Lfeature[2]
        j = 0
        for Rfeature in Rfeatures:
            xr = Rfeature[0]
            yr = Rfeature[1]
            Rr = Rfeature[2]
            if xl >= xr and np.abs(yr -yl) < YTHRESH and np.abs((Rr - Rl)/Rl) < RTHRESH:
                corr = calc_corr(xl, yl, xr, yr, Limg, Rimg)
                if (np.abs(corr) > CORRTHRESH):
                    C_mat[i,j] = np.abs(calc_corr(xl, yl, xr, yr, Limg, Rimg))
            j+=1
        i+=1
    return C_mat


def get_matches(Limg, Rimg, Lfeatures, Rfeatures):

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

    return matchesL
