from .utils import to_grayscale

from numpy import array
from numpy import cross
from numpy.linalg import norm
from numpy import sqrt
from numpy import stack
from numpy import append

from cv2 import composeRT

from json import load

from cv2 import getPerspectiveTransform
from cv2 import warpPerspective
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
    
    leftEyeCircle = list(range(42, 48))
    rightEyeCircle = list(range(36, 42))
    
    # eye corners
    rightEyeOuterCorner = 36
    rightEyeInnerCorner = 39

    leftEyeOuterCorner = 45
    leftEyeInnerCorner = 42

    rightUpperEyeLid = [37, 38]
    rightLowerEyeLid = [40, 41]

    leftUpperEyeLid = [43, 44]
    leftLowerEyeLid = [46, 47]

    leftEyeCorners = [leftEyeInnerCorner, leftEyeOuterCorner]
    leftEyeLid = leftUpperEyeLid+leftLowerEyeLid

    rightEyeCorners = [rightEyeOuterCorner, rightEyeInnerCorner]
    rightEyeLid = rightUpperEyeLid+rightLowerEyeLid

    upperEyeLids = rightUpperEyeLid+leftUpperEyeLid
    lowerEyeLids = rightLowerEyeLid+leftLowerEyeLid

    eyeCorners = rightEyeCorners+leftEyeCorners
    eyeCircles = rightEyeCircle+leftEyeCircle
    eyeLids = rightEyeLid+leftEyeLid

    # mouth corners
    rightMouthCorner = 48
    leftMouthCorner = 54

    mouthCorners = rightMouthCorner+leftMouthCorner

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
        self.size = array([30, 18])
        self.points = array([[0, 9],
                             [30, 9],
                             [15, 0],
                             [15, 18]], dtype='float64')

    def _extract_model_landmarks(self, landmarks):
        return landmarks[self.generic_face_idx].astype('float64')

    def find_extrinsic(self, landmarks, camera):
        _, rotation, translation = solvePnP(objectPoints=self.model_points,
                                            imagePoints=self._extract_model_landmarks(landmarks),
                                            cameraMatrix=camera.matrix,
                                            distCoeffs=camera.distortion,
                                            flags=SOLVEPNP_ITERATIVE)

        rotation, translation = composeRT(rotation, translation, camera.rotation.T, camera.translation.T)[:2]

        return rotation, translation

    def face_model_to_origin(self, landmarks, camera):
        rotation_vector, translation_vector = self.find_extrinsic(landmarks, camera)
        rotation_matrix = Rodrigues(rotation_vector)[0]

        vectors = (rotation_matrix @ self.model_points.T + translation_vector).T

        return camera.vectors_to_origin(vectors)

    @staticmethod
    def extract_rectangle(image, rect, togray=False):
        (x1, y1), (x2, y2) = rect
        extracted_rectangle = image[y1:y2, x1:x2]
        if not togray:
            return extracted_rectangle
        else:
            return to_grayscale(extracted_rectangle)

    @staticmethod
    def get_eye_points(landmarks, stacked=True):
        # axes description: eyes, corners, (x, y)
        corners = landmarks[DlibIdx.eyeCorners].reshape(2, 2, 2)

        # axes description: eyes, (upper, lower), (outer, inner), (x, y)
        # output axes description: eyes, (upper, lower), (x, y)
        lids = landmarks[DlibIdx.eyeLids].reshape(2, 2, 2, 2).mean(axis=2)

        # axes description: eyes, (upper, lower, outer, inner), (x, y)
        if stacked:
            return stack((corners, lids), axis=1)
        else:
            return corners, lids

    @staticmethod
    def get_eye_roi(landmarks):
        # eye center method
        corners = landmarks[DlibIdx.eyeCorners].reshape(2, 2, 2)

        width = corners.ptp(axis=1).reshape(2, 2)[:, 0]
        height = width * 0.4

        roi_size = stack((width, height)).T

        new_roi_size = roi_size / 2

        centers = corners.mean(axis=1)
        centers[:, 1] -= height * 0.1
        eye_roi = stack((centers - new_roi_size, centers + new_roi_size), axis=1).astype(int)

        # min max method
        # eyes = landmarks[eyeCircles].reshape(2, -1, 2)
        # pad = (eyes[:, :, 1].ptp(axis=-1) * pad).astype(int)
        # eye_roi = stack((eyes.min(axis=1) - pad, eyes.max(axis=1) + pad), axis=1)
        return eye_roi, centers, roi_size

    def get_eyes(self, image, landmarks):
        # get points
        corners, lids = self.get_eye_points(landmarks, stacked=True).astype('float64')
        eye_roi, centers, roi_size = self.get_eye_roi(corners)
        print(eye_roi)
        points = append(corners, lids, axis=1)
        for i in range(2):

            eye_image = self.extract_rectangle(image, eye_roi[i])

            # get perspective transform matrix
            print(points[i].reshape(-1, 2))
            M = getPerspectiveTransform(points[i].reshape(-1, 2), self.points)

            eye_image = warpPerspective(eye_image, M, tuple(self.size))
            yield eye_image

        # get eye images
        # eye_roi, centers, roi_size = self.get_eye_roi(points[:, 2:, :])
        # eye_images = [self.extract_rectangle(image, rectangle) for rectangle in eye_roi]


    def extract_eyes(self, image, eye_roi, togray=False):
        return tuple(resize(self.extract_rectangle(image, roi, togray), tuple(self.size)) for roi in eye_roi)



