import numpy as np
import cv2

from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import COLOR_BGRA2GRAY


def calc_norm_disp(x, y):
    dy = np.tile(np.abs(np.arange(y) - np.arange(y).reshape(-1, 1)).reshape(-1, 1, y, 1), (1, x, 1, x))
    dx = np.tile(np.abs(np.arange(x) - np.arange(x).reshape(-1, 1)).reshape(1, -1, 1, x), (y, 1, y, 1))
    disp = np.stack((dx, dy), axis=-1)
    return disp / np.linalg.norm(disp, axis=-1)[:, :, :, :, np.newaxis]


def to_grayscale(image):
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        channels = image.shape[-1]
        if channels == 3:
            return cvtColor(image, COLOR_BGR2GRAY)
        elif channels == 4:
            return cvtColor(image, COLOR_BGRA2GRAY)


def vecs2angles(vectors):
    """
    theta = asin(-y) -- pitch
    phi = atan2(-x, -z) -- yaw
    """
    x, y, z = vectors.T

    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)

    return np.column_stack((yaw, pitch))


def angles2vecs(angles):
    """
    x = (-1) * cos(pitch) * sin(yaw)
    y = (-1) * sin(pitch)
    z = (-1) * cos(pitch) * cos(yaw)
    """
    yaw, pitch = angles.T

    x = (-1) * np.cos(pitch) * np.sin(yaw)
    y = (-1) * np.sin(pitch)
    z = (-1) * np.cos(pitch) * np.cos(yaw)

    vectors = np.column_stack((x, y, z))
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / norm

def detect_pupil(image):

    image = to_grayscale(image)

    image = cv2.blur(image, (5, 5))

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # print(sobelx.shape, sobely.shape)
    sobel = np.stack((sobelx, sobely), axis=-1)
    # print(sobel.shape)
    mag = np.linalg.norm(sobel, axis=-1)
    mag[mag < (0.3 * mag.std() + mag.std())] = np.nan
    norm_sobel = sobel / mag[:, :, np.newaxis]

    norm_disp = calc_norm_disp(*image.shape[::-1])
    # print(norm_disp.shape)
    projection_matr = norm_disp * sobel
    # projection_matr = ((self.norm_disp_x * norm_sobelx + self.norm_disp_y * norm_sobely))
    projection_matr[projection_matr < 0] = np.nan
    print(projection_matr.shape)
    ind = np.argmax(np.nanmean(projection_matr, axis=(2, 3)) * (1 - image / 255)[:, :, np.newaxis])
    return np.unravel_index(ind, image.shape)[::-1]
