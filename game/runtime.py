from game.estimator import *

from game.dev import Device
from game.dev import Camera
from game.dev import Picture
from game.dev import WebCameraName
from game.dev import InfraredCameraName
from game.dev import KinectColorName
from game.dev import KinectInfraredName
from game.dev import load_devices

from game.estimator.utils import detect_pupil

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


def ispressed(key, delay=None):
    return cv2.waitKey(delay=delay) == key


class GameRuntime(object):

    def __init__(self, face_detector_config, landmarks_handler_config, load_config):

        # load cameras
        load_devices(**load_config)

        # assign camera objects
        self._kinect_color = Camera.get(KinectColorName)
        self._web = Camera.get(WebCameraName)
        self._ir = Camera.get(InfraredCameraName)
        self._kinect_ir = Camera.get(KinectInfraredName)

        self._ir.change_properties(ExposureTime=80000, GainAuto='Off', ExposureAuto='Off')

        # assing cameras to handle
        self._cams = cycle([cam for cam in self.all_cams if cam.connected])

        self._curr_cam = next(self._cams)
        self._curr_frame = None

        # get interaction devices
        self._pictures = [pic for pic in Picture.values()]
        self._pic_translations = np.array([pic.translation.flatten() for pic in self._pictures])

        # init face handling models

        # load gazenet
        # self._gazenet = GazeNet().load_weigths(gaze_model_path)
        # self._eye_image_shape = tuple(self._gazenet.image_shape[::-1])

        # load face detector
        self._face_detector = FaceDetector(**face_detector_config)

        # load landmarks handler
        self._landmarks_handler = LandmarksHandler(**landmarks_handler_config)

        # face stack
        self._face = []

        # frames check queue
        self._last_frames = []

        # here we will store skeleton data
        self._bodies = None

        self.landmarks_stack = []

    @property
    def all_cams(self):
        return [self._ir, self._kinect_color, self._web]

    def next_cam(self):
        cv2.destroyWindow('picture')
        self._curr_cam = next(self._cams)
        self._last_frames.clear()
        self._curr_cam.restart()

    def draw_landmarks(self, landmarks, color=None, radius=2):
        color = color if color is not None else (255, 255, 255)
        for point in landmarks.astype(int).tolist():
            cv2.circle(self._curr_frame, tuple(point), radius, color, -1)

    def draw_gaze(self, face_gaze_line_points, color=(255, 0, 0)):
        start_pos, end_pos = map(tuple, face_gaze_line_points.astype(int))
        cv2.line(self._curr_frame, start_pos, end_pos, color, 4)

    @staticmethod
    def check_origin_face_coordinates(face_gaze_line_points, landmarks, tol=15):
        return np.abs(face_gaze_line_points[0] - landmarks[NoseTip]).sum() < tol

    def clear_frame(self):
        self._curr_frame = None

    def find_attention(self, gaze, threshold=0.15):

        ground_truth = self._pic_translations - gaze.line[0]
        norm_ground_truth = np.linalg.norm(ground_truth, axis=1, keepdims=True)
        ground_truth = ground_truth / norm_ground_truth
        attentions = np.linalg.norm((ground_truth - gaze.vector) * np.array([1, 0.7, 1]), axis=1)

        pic_index = attentions.argmin()

        if attentions[pic_index] < threshold:
            return pic_index
        else:
            return None

    def handle_face(self, landmarks):

        # get landmarks in 3d
        origin_landmarks = self._landmarks_handler.face_model_to_origin(landmarks, self._curr_cam)
        landmarks = landmarks.astype(int)
        # calculate face gaze
        face_gaze = self._landmarks_handler.calc_face_gaze(origin_landmarks)

        # get origin point of nose
        origin_nose = face_gaze.line[0]

        # project face gaze
        face_gaze_line_2d = self._kinect_color.project_points(face_gaze.line)

        # extract eyes
        eye_pts1, eye_images = self._landmarks_handler.extract_eyes(self._curr_frame, landmarks, pad=0.2)
        # for i, eye_image in enumerate(eye_images):
        #     cv2.imshow(f'Eye {i}', cv2.resize(eye_image, (0, 0), fx=5, fy=5))

        # pupil_centers = eye_pts1 + np.array(list((detect_pupil(eye_image, i) for i, eye_image in enumerate(eye_images))))

        # self.draw_landmarks(eye_pts1, color=(0, 0, 255), radius=2)
        self.draw_landmarks(landmarks)
        # self.draw_landmarks(pupil_centers, color=(0, 255, 0), radius=2)

        # check origin points and draw
        if self.check_origin_face_coordinates(face_gaze_line_2d, landmarks):

            attention_pic_index = self.find_attention(face_gaze)

            if attention_pic_index is not None:
                self._pictures[attention_pic_index].show_pic(winname='picture')
            else:
                cv2.destroyWindow('picture')

            self.draw_gaze(face_gaze_line_2d)

        # elif :
            # cv2.destroyWindow('picture')

    def check_face(self):
        enough = len(self._face) >= 3
        if enough:
            self._face.pop(0)
        return enough

    def handle_frame(self):
        assert self._curr_frame is not None

        _, faces = self._face_detector.extract_faces(self._curr_frame)

        if faces:
            self._face.append(faces[0])
        if self.check_face():
            self.handle_face(np.array(self._face[:]).mean(axis=0))
            self._last_frames.append(True)
        elif not faces:
            self._last_frames.append(False)

    def show(self):
        cv2.imshow('window', cv2.resize(self._curr_frame, (1280, 920)))
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
            if self._curr_cam.name == 'WebCamera':
                # print(self._curr_frame.shape)
                self._curr_frame = cv2.resize(self._curr_frame, (0, 0), fx=2, fy=2)
            self.handle_frame()
            self.show()
        except RuntimeError:
            self._curr_cam.restart()
            pass
        except AssertionError:
            pass
        finally:
            self.check_frames()

    def run(self):
        # -------- Main Program Loop -----------
        while not ispressed(27, delay=1):
            self.tick()

        cv2.destroyAllWindows()
