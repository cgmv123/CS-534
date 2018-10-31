import cv2
import numpy as np

reprojThresh=5

def get_stitched_image(img1, img2, H):
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, H)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(H) ,(x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] = img1

    # Return the result
    return result_img

def stitch(matches, Limg, Rimg, Lfeatures, Rfeatures):
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

    (H, status) = cv2.findHomography(np.asarray(ptsL), np.asarray(ptsR), cv2.RANSAC, reprojThresh)

    return get_stitched_image(Rimg, Limg, H)