import cv2
import os


def load_files(dir):
    imgs = []
    for file in sorted(os.listdir(dir)):
        if file.endswith(".png"):
            print(file)
            imgs.append(cv2.imread(dir + "/" + file, 1))

    imgs.reverse()
    return imgs


def frames(video):
    cap = cv2.VideoCapture(video)

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    frame_interval = 0
    frame_num = 1
    output = []
    minp = p0[0][0][0]

    j = 0
    for p in p0:
        if p[0][0] < minp:
            minp = p[0][0]
            pos = j
        j = j + 1

    left_to_right = True
    # determine frame interval
    while frame_interval == 0:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_num = frame_num + 1
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]

        if abs(good_new[pos][0] - minp) / width > 0.5:
            frame_interval = frame_num
            if good_new[pos][0] - minp > 0:
                left_to_right = False

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # collect frames
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    while cap.isOpened():
        if left_to_right:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_interval * i)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - (frame_interval * i) - 1)
        ret, frame = cap.read()
        if ret == True & ((frame_count - (frame_interval * i) - 1) > 0):
            output.append(frame)
            # cv2.imwrite('file'+str(i)+'.jpg', output[i])
        else:
            if left_to_right:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                # print(i)
                ret, frame = cap.read()
                output.append(frame)
                # cv2.imwrite('file'+str(i)+'.jpg', output[i])
            break
        i = i + 1
    return output
