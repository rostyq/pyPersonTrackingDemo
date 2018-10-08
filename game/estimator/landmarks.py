from .utils import to_grayscale

from numpy import array
from numpy import cross
from numpy.linalg import norm
from numpy import sqrt
from numpy import stack

from json import load

from cv2 import resize
from cv2 import Rodrigues
from cv2 import solvePnP
from cv2 import SOLVEPNP_ITERATIVE
from collections import namedtuple

flip_array = array([1, -1, 1])

Gaze = namedtuple('Gaze', 'vector line')


class DlibIdx:

    # face model indices
    noseTip = 30
    chin = 8

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

    landmarks_to_model = [noseTip, chin, rightEyeOuterCorner, leftEyeOuterCorner, rightMouthCorner, leftMouthCorner]


class LandmarksHandler:

    noseTip = 0
    chin = 1
    rightEyeOuterCorner = 2
    leftEyeOuterCorner = 3
    leftMouthCorner = 4
    rightMouthCorner = 5

    def __init__(self, path_to_face_model, chin_nose_distance):

        # self.model_points = loadmat(path_to_face_model)['model'] * flip_array
        with open(path_to_face_model, 'r') as f:
            generic_face_data = load(f)

        self.generic_face_keys = list(generic_face_data.keys())
        self.generic_face_idx = [getattr(DlibIdx, key) for key in self.generic_face_keys]

        for i, key in enumerate(self.generic_face_keys):
            setattr(self, key, i)

        self.model_points = array(list(generic_face_data.values())) * flip_array
        self.chin_nose_distance = chin_nose_distance
        self.face_scale = self.chin_nose_distance / sqrt((self.model_points[self.chin] ** 2).sum())
        self.model_points = self.model_points * self.face_scale
        self.roi_size = array([30, 18])

    def _extract_model_landmarks(self, landmarks):
        return landmarks[self.generic_face_idx].astype('float64')

    def find_extrinsic(self, landmarks, matrix, distortion):
        _, rotation_vector, translation_vector = solvePnP(objectPoints=self.model_points,
                                                          imagePoints=self._extract_model_landmarks(landmarks),
                                                          cameraMatrix=matrix,
                                                          distCoeffs=distortion,
                                                          flags=SOLVEPNP_ITERATIVE)

        return rotation_vector, translation_vector

    def face_model_to_origin(self, landmarks, camera):
        rotation_vector, translation_vector = self.find_extrinsic(landmarks, camera.matrix, camera.distortion)
        rotation_matrix = Rodrigues(rotation_vector)[0]

        vectors = (rotation_matrix @ self.model_points.T + translation_vector).T

        print(f'before {vectors[0]}')
        print(f'after {camera.vectors_to_origin(vectors)[0]}')

        return camera.vectors_to_origin(vectors)

    def check_origin_face_coordinates(self, face_gaze_line_points, landmarks, tol=15):
        return abs(face_gaze_line_points[0] - landmarks[DlibIdx.noseTip]).sum() < tol

    @staticmethod
    def calc_face_normal(pt1, pt2, pt3):
        cross_vec = cross(pt1 - pt2, pt1 - pt3)
        return cross_vec / norm(cross_vec)

    @staticmethod
    def calc_gaze_line(gaze, origin_point, coeff=0.2):
        face_enpoint = origin_point + gaze * coeff
        return stack((origin_point, face_enpoint))

    def calc_face_gaze(self, origin_landmarks, **kwargs):
        chin_point = origin_landmarks[self.chin]
        eye_corners = origin_landmarks[[self.leftEyeOuterCorner, self.rightEyeOuterCorner]]
        vector = self.calc_face_normal(chin_point, *eye_corners)
        line = self.calc_gaze_line(gaze=vector, origin_point=origin_landmarks[self.noseTip], **kwargs)
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
        eye_corners = landmarks[DlibIdx.rightEyeCorners + DlibIdx.leftEyeCorners].reshape(2, 2, 2)
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
