import cv2
import numpy as np


def place_image(base, img):
    h, w = img.shape[:2]
    dest_slice = np.s_[0:h, 0:w]
    dest = base[dest_slice]
    mask = (255 - img[..., 3])
    dest_bg = cv2.bitwise_and(dest, dest, mask=mask)
    dest = cv2.add(dest_bg, img)
    base[dest_slice] = dest


def findDimensions(img, H):
    w2, h2 = img.shape[:2]
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)
    img_dims = cv2.perspectiveTransform(img2_dims_temp, H)
    [min_x, min_y] = np.int32(img_dims.min(axis=0).ravel() - 0.5)
    [max_x, max_y] = np.int32(img_dims.max(axis=0).ravel() + 0.5)

    return min_x, min_y, max_x, max_y


def stitch(imgs, H_map):
    tot_min_x = np.inf
    tot_min_y = np.inf
    tot_max_x = -np.inf
    tot_max_y = -np.inf

    num_images = len(imgs)

    for i in range(0, len(H_map)):
        curr_img = imgs[i]
        (min_x, min_y, max_x, max_y) = findDimensions(curr_img, (H_map[i]))
        tot_min_x = np.floor(np.minimum(min_x, tot_min_x))
        tot_min_y = np.floor(np.minimum(min_y, tot_min_y))
        tot_max_x = np.ceil(np.maximum(max_x, tot_max_x))
        tot_max_y = np.ceil(np.maximum(max_y, tot_max_y))

    pan_height = tot_max_y - tot_min_y
    pan_width = tot_max_x - tot_min_x

    pan_size = (int(pan_height), int(pan_width))

    final_img = np.zeros((pan_size[0], pan_size[1], 4)).astype("uint8")

    # warp images
    warped = []

    for i in range(0, num_images):
        img = imgs[i]

        (min_x, min_y, max_x, max_y) = findDimensions(img, (H_map[i]))

        max_x = max(tot_max_x, max_x)
        max_y = max(tot_max_y, max_y)
        min_x = min(tot_min_x, min_x)
        min_y = min(tot_min_y, min_y)

        curr_width = max_x - min_x
        curr_height = max_y - min_y

        curr_size = (int(curr_width), int(curr_height))

        new_img = cv2.warpPerspective(imgs[i], (H_map[i]), curr_size)
        new_img = new_img.astype("uint8")
        place_image(final_img, new_img)
        warped.append(new_img)

    return final_img
