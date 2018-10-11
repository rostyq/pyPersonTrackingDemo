from game.estimator import *
from time import sleep
from game.dev import Device
from game.dev import Camera
from game.dev import Picture
from game.dev import WebCameraName
from game.dev import InfraredCameraName
from game.dev import KinectColorName
from game.dev import KinectInfraredName
from game.dev import load_devices

from game.estimator.utils import detect_pupil
from game.estimator.utils import compose_rotation_matrix
from game.estimator.utils import decompose_rotation_matrix

import ctypes
import pygame
import sys

from itertools import cycle

import numpy as np
from scipy.stats import mode
import cv2

if sys.hexversion >= 0x03000000:
    pass
else:
    pass

# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]


def joint_to_np(joint):
    position = joint.Position
    x = position.x
    y = position.y
    z = position.z
    return np.array([x, y, z])


def ispressed(key, delay=None):
    return cv2.waitKey(delay=delay) == key


class Face(Device):

    null_point = array([0, 0, 0])

    def __init__(self, index=0, name='Face', translation=None, rotation=None):
        super().__init__(index=index, name=name, translation=translation, rotation=rotation)
        self.face_line = None
        self.gaze_line = None

    def get_face_line(self, coeff=0.1):
        return np.stack((self.null_point,
                         self.normal.flatten()*coeff)) + self.translation.flatten()

    def get_gaze_line(self, yaw, pitch, roll, coeff=0.1):
        return np.stack((self.null_point,
                         self.calc_gaze_vector(yaw,
                                               pitch,
                                               roll).flatten()*coeff)) + self.translation.flatten()

    def update_face_line(self, **kwargs):
        self.face_line = self.get_face_line(**kwargs)

    def rotate_self_normal(self, yaw, pitch, roll):
        rotation_matrix = compose_rotation_matrix(yaw, pitch, roll)
        return rotation_matrix @ self._self_normal

    def calc_gaze_vector(self, yaw, pitch, roll):
        return self.to_world(self.rotate_self_normal(yaw, pitch, roll), translate=False)

    def update_gaze_line(self, yaw, pitch, roll, **kwargs):
        self.gaze_line = self.get_gaze_line(yaw, pitch, roll, **kwargs)


class GameRuntime(object):

    def __init__(self, face_detector_config: dict, landmarks_handler_config: dict, load_config: dict):

        # load cameras
        load_devices(**load_config)

        # assign camera objects
        self._kinect_color = Camera.get(KinectColorName)
        self._web = Camera.get(WebCameraName)
        self._ir = Camera.get(InfraredCameraName)
        self._kinect_ir = Camera.get(KinectInfraredName)

        self._ir.change_properties(ExposureTime=60000, GainAuto='Off', Gain=8.0, ExposureAuto='Off', ReverseX=False)

        # assing cameras to handle
        self._cams = cycle([cam for cam in self.all_cams if cam.connected])

        self._curr_cam = next(self._cams)
        self._curr_frame = None

        # get interaction devices
        self._pictures = [pic for pic in Picture.values()]
        self._pic_translations = np.array([pic.translation.flatten() for pic in self._pictures])

        # init face handling models

        # load face detector
        self._face_detector = FaceDetector(**face_detector_config)

        # load landmarks handler
        self._landmarks_handler = LandmarksHandler(**landmarks_handler_config)

        # face stack
        self._last_landmarks = []

        # frames check queue
        self._last_frames = []
        self._last_pic_indices = []

        # here we will store skeleton data
        self._bodies = None

        self._curr_face = Face()

        self.landmarks_stack = []

    @property
    def all_cams(self):
        return [self._ir, self._kinect_color, self._web]

    def next_cam(self):
        cv2.destroyWindow('picture')
        self._curr_cam.stop()
        self._curr_cam = next(self._cams)
        self._last_frames.clear()
        self._curr_cam.start()

    def draw_landmarks(self, landmarks, color=None, radius=2):
        color = color if color is not None else (255, 255, 255)
        for point in landmarks.astype(int).tolist():
            cv2.circle(self._curr_frame, tuple(point), radius, color, -1)

    def draw_rectangle(self, rectangle, color=None, thickness=1):
        rectangle = rectangle.astype(int)
        pt1 = tuple(rectangle[0])
        pt2 = tuple(rectangle[1])
        color = color if color else (0, 255, 0)
        cv2.rectangle(self._curr_frame, pt1, pt2, color=color, thickness=thickness)

    def draw_rectangles(self, rectangles, colors=None, thickness=2):
        colors = colors if colors is not None else SKELETON_COLORS
        for (x, y, w, h), color in zip(rectangles, colors):
            cv2.rectangle(self._curr_frame, (x, y), (x+w, y+h), color, thickness=thickness)

    def draw_gaze(self, line_points, color=(255, 0, 0)):
        start_pos, end_pos = map(tuple, line_points.astype(int))
        cv2.line(self._curr_frame, start_pos, end_pos, color, 4)

    def clear_frame(self):
        self._curr_frame = None

    def find_attention(self, threshold=0.7):

        ground_truth = self._pic_translations - self._curr_face.face_line[0]
        norm_ground_truth = np.linalg.norm(ground_truth, axis=1, keepdims=True)
        ground_truth = ground_truth / norm_ground_truth
        # attentions = np.linalg.norm((ground_truth - self._curr_face.normal) * np.array([1, 0.6, 1]), axis=1)
        attentions = ground_truth @ self._curr_face.normal.T
        # attentions = attentions[attentions > 0.0]

        pic_index = attentions.argmax()

        if attentions[pic_index] > threshold:
            return pic_index
        else:
            return None

    @staticmethod
    def check_face_extrinsic(proj_nose_point, nose_point, tol=15):
        return abs(proj_nose_point - nose_point).sum() < tol

    def update_face(self, landmarks):
        rotation, translation = self._landmarks_handler.find_extrinsic(landmarks, self._curr_cam)
        self._curr_face.rotation = rotation
        self._curr_face.translation = translation
        self._curr_face.update_face_line()

    def handle_face(self):

        # update face
        self.update_face(np.array(self._last_landmarks[:]).mean(axis=0))

        # change landmarks type
        landmarks = self._last_landmarks[-1]

        # calculate face gaze 2d points
        face_gaze_line_2d = self._curr_cam.project_points(self._curr_face.face_line)

        # get eye regions of interest
        # corners, lids = self._landmarks_handler.get_eye_points(landmarks)
        eye_roi, eye_centers, roi_size = self._landmarks_handler.get_eye_roi(landmarks)

        self.draw_landmarks(eye_roi.reshape(-1, 2))

        # eye_points = self._landmarks_handler.get_eye_points(landmarks)
        # print(eye_points)
        # self.draw_landmarks(eye_points.reshape(-1, 2))
        #
        # get eyes
        # eye_images = self._landmarks_handler.extract_eyes(self._curr_frame, eye_roi)
        # eye_images = self._landmarks_handler.get_eyes(self._curr_frame, landmarks)

        # estimate pupil centers
        # pupil_fraction = np.array([detect_pupil(eye_image, i) for i, eye_image in enumerate(eye_images)])
        # pupil_centers = pupil_fraction * roi_size
        # pupil_landmarks = eye_roi[:, 0, :] + pupil_centers

        # print(pupil_landmarks)
        #
        # self.draw_landmarks(pupil_landmarks, color=(0, 255, 0), radius=2)

        # for i, eye_image in enumerate(eye_images):
        #     key = 'right' if not i else 'left'
        #     cv2.imshow(f'Eye {key}', cv2.resize(eye_image, (0, 0), fx=8, fy=8))

        # for eye_region in eye_roi:
        #     self.draw_rectangle(eye_region)

        self.draw_landmarks(landmarks)

        # check origin points and draw
        if self.check_face_extrinsic(face_gaze_line_2d[0], landmarks[30], tol=30):

            attention_pic_index = self.find_attention()

            if attention_pic_index is not None:
                self._last_pic_indices.append(attention_pic_index)
                average_index = int(mode(self._last_pic_indices[:])[0])
                self._pictures[average_index].show_pic(winname='picture')
            else:
                cv2.destroyWindow('picture')

            self.draw_gaze(face_gaze_line_2d)

        # elif :
            # cv2.destroyWindow('picture')

    def check_indices(self):
        if len(self._last_pic_indices) >= 3:
            self._last_pic_indices.pop(0)
            return True
        else:
            return False

    def check_face(self):
        if len(self._last_landmarks) >= 5:
            self._last_landmarks.pop(0)
            return True
        else:
            return False

    def handle_frame(self):
        assert self._curr_frame is not None

        rectangles, faces = self._face_detector.extract_faces(self._curr_frame,
                                                              **self._curr_cam.face_detect_kwargs)

        if faces:
            self._last_landmarks.append(faces[0])
            self.draw_rectangles(rectangles)
            if self.check_face():
                self.handle_face()
                self._last_frames.append(True)
        elif not faces:
            self._last_frames.append(False)
            sleep(1/15)


    def show(self):
        cv2.imshow('window', cv2.resize(self._curr_frame, (1280, 920)))
        # cv2.imshow('window', self._curr_frame)
        self.clear_frame()

    def update_frame(self):
        self._curr_frame = self._curr_cam.get_frame()

    def check_frames(self):
        if (len(self._last_frames) == 10) and (not any(self._last_frames)):
            self.next_cam()
        if len(self._last_frames) >= 10:
            self._last_frames.pop(0)

    def tick(self):

        try:
            # get frame
            self.update_frame()
            self.handle_frame()
            self.show()
        except RuntimeError:
            self._curr_cam.restart()
            pass
        except AssertionError as e:
            # print(e)
            pass
        finally:
            self.check_frames()
            self.check_indices()

    def run(self):
        # -------- Main Program Loop -----------
        while not ispressed(27, delay=1):
            self.tick()

        cv2.destroyAllWindows()
