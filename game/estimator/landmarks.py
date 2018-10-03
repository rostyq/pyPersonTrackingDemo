from .utils import to_grayscale

from scipy.io import loadmat
from numpy import array
from numpy import cross
from numpy.linalg import norm
from numpy import sqrt
from numpy import stack
from numpy import diff
from numpy import abs as np_abs

from cv2 import resize
from cv2 import Rodrigues
from cv2 import solvePnP
from cv2 import SOLVEPNP_ITERATIVE
from cv2 import INTER_CUBIC

from collections import namedtuple

flip_array = array([-1, -1, 1])

# face model indices
NoseTip = 30
Chin = 8

# eye corners
rightEyeOuterCorner = 36
rightEyeInnerCorner = 39

leftEyeOuterCorner = 45
leftEyeInnerCorner = 42

leftEyeCorners = [leftEyeOuterCorner, leftEyeInnerCorner]
rightEyeCorners = [rightEyeInnerCorner, rightEyeOuterCorner]
eyeCorners = leftEyeCorners+rightEyeCorners

leftEyeCircle = list(range(42, 48))
rightEyeCircle = list(range(36, 42))

eyeCircles = leftEyeCircle+rightEyeCircle

# mouth corners
rightMouthCorner = 48
leftMouthCorner = 54

landmarks_to_model = [NoseTip, Chin, rightEyeOuterCorner, leftEyeOuterCorner, rightMouthCorner, leftMouthCorner]

originNoseTip = 0
originChin = 1
originRightEyeOuterCorner = 2
originLeftEyeOuterCorner = 3
originRightMouthCorner = 4
originLeftMouthCorner = 5

Gaze = namedtuple('Gaze', 'vector line')


class LandmarksHandler:

    def __init__(self, path_to_face_model, chin_nose_distance):

        self.model_points = loadmat(path_to_face_model)['model'] * flip_array
        self.chin_nose_distance = chin_nose_distance
        self.face_scale = self.chin_nose_distance / sqrt((self.model_points[1] ** 2).sum())
        self.model_points = self.model_points * self.face_scale
        self.roi_size = array([30, 18])

    @staticmethod
    def _extract_model_landmarks(landmarks):
        return landmarks[landmarks_to_model].astype('float64')

    def find_extrinsic(self, landmarks, matrix, distortion):
        _, rotation_vector, translation_vector = solvePnP(self.model_points,
                                                          self._extract_model_landmarks(landmarks),
                                                          matrix,
                                                          distortion,
                                                          flags=SOLVEPNP_ITERATIVE)

        return rotation_vector, translation_vector

    def face_model_to_origin(self, landmarks, camera):
        rotation_vector, translation_vector = self.find_extrinsic(landmarks, camera.matrix, camera.distortion)
        rotation_matrix = Rodrigues(rotation_vector)[0]

        vectors = rotation_matrix @ self.model_points.T + translation_vector
        return camera.vectors_to_origin(vectors.T)

    @staticmethod
    def calc_face_normal(pt1, pt2, pt3):
        cross_vec = cross(pt1 - pt2, pt1 - pt3)
        return cross_vec / norm(cross_vec)

    @staticmethod
    def calc_gaze_line(gaze, origin_point, coeff=0.2):
        face_enpoint = origin_point + gaze * coeff
        return stack((origin_point, face_enpoint))

    def calc_face_gaze(self, origin_landmarks, **kwargs):
        chin_point = origin_landmarks[originChin]
        eye_corners = origin_landmarks[[originLeftEyeOuterCorner, originRightEyeOuterCorner]]
        vector = self.calc_face_normal(chin_point, *eye_corners)
        line = self.calc_gaze_line(gaze=vector, origin_point=origin_landmarks[originNoseTip], **kwargs)
        return Gaze(vector=vector, line=line)

    @staticmethod
    def extract_rectangle(image, rect, togray=False):
        (x1, y1), (x2, y2) = rect
        extracted_rectangle = image[y1:y2, x1:x2]
        if not togray:
            return extracted_rectangle
        else:
            return to_grayscale(extracted_rectangle)

    def extract_eyes(self, image, landmarks, togray=False, pad=0.5):

        # eye center method
        eye_corners = landmarks[rightEyeCorners + leftEyeCorners].reshape(2, 2, 2)
        eye_width = eye_corners.ptp(axis=1).reshape(2, 2)[:, 0]
        eye_height = eye_width * 0.6

        new_roi_size = (stack((eye_width, eye_height)) / 2).T

        eye_centers = eye_corners.mean(axis=1)
        eye_centers[:, 1] -= eye_height * 0.1
        eye_roi = stack((eye_centers - new_roi_size, eye_centers + new_roi_size), axis=1).astype(int)

        # min max method
        # eyes = landmarks[eyeCircles].reshape(2, -1, 2)
        # pad = (eyes[:, :, 1].ptp(axis=-1) * pad).astype(int)
        # eye_roi = stack((eyes.min(axis=1) - pad, eyes.max(axis=1) + pad), axis=1)
        return eye_roi[:, 0], (resize(self.extract_rectangle(image,
                                                             roi,
                                                             togray),
                                      tuple(self.roi_size)) for roi in eye_roi)
