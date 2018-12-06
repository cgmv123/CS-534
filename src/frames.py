import cv2 as cv
import numpy as np

sift = cv2.SIFT()

def get_frames(video):
    cap = cv.VideoCapture(video)
    nextFrame = 0

    while cap.isOpened():
        suc, next = cap.read()
        if suc:
            frames[nextFrame] = next
            nextFrame += 1

    for x in range(0, nextFrame):
       output =  main(output, frames [x])

    return output