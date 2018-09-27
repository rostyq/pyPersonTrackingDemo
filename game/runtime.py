from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

from game.estimator import *
from game.estimator import LandmarksHandler

from game.dev import Camera
from game.dev import defaultWebCameraName
from game.dev import defaultInfraredCameraName
from game.dev import defaultKinectColorName
from game.dev import defaultKinectInfraredName

import ctypes
import pygame
import sys

from itertools import cycle

import numpy as np
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


class GameRuntime(object):

    def __init__(self, face_detector_config, landmarks_handler_config, gaze_model_path):

        # init face handling models

        # load gazenet
        self._gazenet = GazeNet().load_weigths(gaze_model_path)
        self._eye_image_shape = tuple(self._gazenet.image_shape[::-1])

        # load face detector
        self._face_detector = FaceDetector(**face_detector_config)

        # load landmarks handler
        self._landmarks_handler = LandmarksHandler(**landmarks_handler_config)

        # assign camera objects
        self._kinect_color = Camera.get(defaultKinectColorName)
        self._web = Camera.get(defaultWebCameraName)
        self._ir = Camera.get(defaultInfraredCameraName)
        self._kinect_ir = Camera.get(defaultKinectInfraredName)

        # assing cameras to handle
        self._cams = cycle([cam for cam in self.all_cams if cam.connected])

        self._curr_cam = next(self._cams)
        self._curr_frame = None

        # counter
        self._last_frames = []

        # here we will store skeleton data
        self._bodies = None

        self.landmarks_stack = []

    @property
    def all_cams(self):
        return [self._ir, self._kinect_color, self._web]

    def next_cam(self):
        # self._curr_cam.stop()
        self._curr_cam = next(self._cams)
        # self._curr_cam.start()
        self._last_frames.clear()

    def draw_landmarks(self, landmarks, color=None):
        color = color if color is not None else (255, 255, 255)
        for point in landmarks.astype(int).tolist():
            cv2.circle(self._curr_frame, tuple(point), 2, color, -1)

    @staticmethod
    def calc_gaze_line(gaze, origin_point, coeff=5):
        face_enpoint = origin_point + gaze / coeff
        return np.stack((origin_point, face_enpoint))

    def draw_gaze(self, face_gaze_line_points, color=(255, 0, 0)):
        start_pos, end_pos = map(tuple, face_gaze_line_points.astype(int))
        cv2.line(self._curr_frame, start_pos, end_pos, color, 4)

    @staticmethod
    def check_origin_face_coordinates(face_gaze_line_points, landmarks, tol=15):
        return np.abs(face_gaze_line_points[0] - landmarks[NoseTip]).sum() < tol

    def clear_frame(self):
        self._curr_frame = None

    def handle_face(self, landmarks):

        self.draw_landmarks(landmarks)

        # get landmarks in 3d
        origin_landmarks = self._landmarks_handler.face_model_to_origin(landmarks,
                                                                        self._curr_cam)

        # calculate face gaze
        face_gaze = self._landmarks_handler.calc_face_gaze(origin_landmarks)

        # get origin point of nose
        origin_nose = origin_landmarks[originNoseTip]

        # calc origin face gaze
        origin_face_gaze_line_points = self.calc_gaze_line(face_gaze, origin_nose)

        # project face gaze
        face_gaze_line_points = self._kinect_color.project_vectors(origin_face_gaze_line_points)

        # check origin points and draw
        if self.check_origin_face_coordinates(face_gaze_line_points, landmarks):
            self.draw_gaze(face_gaze_line_points)

    def handle_frame(self):
        assert self._curr_frame is not None

        _, faces = self._face_detector.extract_faces(self._curr_frame)

        if faces:
            self.handle_face(faces[0])
            self._last_frames.append(True)
        elif not faces:
            self._last_frames.append(False)

    def show(self):
        cv2.imshow('window', cv2.resize(self._curr_frame, (1280, 920)))
        self.clear_frame()

    def run(self):
        # -------- Main Program Loop -----------
        while not cv2.waitKey(1) == 27:

            # get frame
            self._curr_frame = self._curr_cam.get_frame()
            if self._curr_cam.name == 'WebCamera':
                self._curr_frame = cv2.resize(self._curr_frame, (0, 0), fx=2, fy=2)
            try:
                self.handle_frame()
                self.show()
            except RuntimeError as e:
                print(e)
                # self._curr_cam.restart()
                pass
            except AssertionError as e:
                print(e)
                # self._curr_cam.restart()
                pass
            finally:
                if (len(self._last_frames) == 10) and (not any(self._last_frames)):
                    self.next_cam()
                if len(self._last_frames) >= 10:
                    self._last_frames.pop(0)
            continue

        cv2.destroyAllWindows()

