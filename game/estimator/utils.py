import numpy as np
import cv2

from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
from cv2 import COLOR_BGRA2GRAY

resize_shape = np.array([15, 9])


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


def detect_pupil(eye_image, index=0):

    def calc_threshold(mag, coeff):
        return coeff * mag.std() + mag.mean()

    def calc_m(disp_x, disp_y, grad_x, grad_y, power=2):
        M_1 = (disp_x ** power * grad_x ** power + disp_y ** power * grad_y ** power)
        M_2 = 2 * disp_x * disp_y * grad_x * grad_y
        M_3 = (disp_x ** power * grad_y ** power + disp_y ** power * grad_x ** power)
        return M_1, M_2, M_3

    # input_shape = np.array(eye_image.shape[:2][::-1])
    #
    # eye_image = cv2.resize(eye_image, (30, 18))

    gray = to_grayscale(eye_image)
    # gray = cv2.equalizeHist(gray)

    height, width = gray.shape

    border = int(height * 0.2)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.copy(np.sqrt(grad_x ** 2 + grad_y ** 2))
    func = np.zeros(eye_image.shape)

    # grad_x[mag < (0.3 * mag.std() + mag.mean())] = np.nan
    # grad_y[mag < (0.3 * mag.std() + mag.mean())] = np.nan

    max_x, max_y, max_val = 0, 0, 0

    disp_x = np.tile(np.arange(width), (height, 1))
    disp_y = np.tile(np.arange(height), (width, 1)).T

    threshold = calc_threshold(mag, 1.5)

    mask_threshold = mag > threshold
    disp_x = disp_x[mask_threshold].reshape(-1)
    disp_y = disp_y[mask_threshold].reshape(-1)
    grad_x = grad_x[mask_threshold].reshape(-1)
    grad_y = grad_y[mask_threshold].reshape(-1)

    for i in range(width):
        for j in range(height):
            M_1, M_2, M_3 = calc_m(disp_x, disp_y, grad_x, grad_y, power=2)
            val = np.nansum((M_1 + M_2) / (M_1 + M_3))
            # val = val * (1 / (1 + (0.1 * (width/2 - i)) ** 2 + (0.1 * (height/2 - j)) ** 2))
            func[j, i] = val
            if val > max_val:
                max_val = val
                max_x, max_y = i, j
            disp_y -= 1
        disp_y = np.tile(np.arange(height), (width, 1)).T
        disp_y = disp_y[mag > threshold].reshape(-1)
        disp_x -= 1
    mag = cv2.normalize(mag, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    func = cv2.normalize(func, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # cv2.imshow(f"Function {index}", cv2.resize(func, (0, 0), fx=5, fy=5))
    # cv2.imshow(f"Gradients {index}", cv2.resize(mag, (0, 0), fx=5, fy=5))
    cv2.circle(eye_image, (max_x, max_y), 1, (0, 0, 255), -1)
    cv2.imshow(f"eye input {index}", cv2.resize(eye_image, (0, 0), fx=8, fy=8))
    return max_x, max_y
    # return (np.array([max_x, max_y]) / resize_shape * input_shape).astype(int)
