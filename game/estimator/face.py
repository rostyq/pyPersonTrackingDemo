from .utils import to_grayscale

from dlib import shape_predictor
from dlib import rectangle as DlibRectangle
from dlib import rectangles as DlibRectangles

from cv2 import CascadeClassifier

from numpy import array


class FaceDetector:

    def __init__(self, path_to_face_points, path_to_hc_model, scale=1, minNeighbors=5):

        # load_weigths face detector model
        self.detector = CascadeClassifier(str(path_to_hc_model)).detectMultiScale

        # parameters for face detector model
        self.scale = scale
        self.minNeighbors = minNeighbors

        # load_weigths face landmarks detector model
        self.predictor = shape_predictor(str(path_to_face_points))

    @staticmethod
    def shape_to_np(shape, dtype='int'):
        return array([[shape.part(i).x, shape.part(i).y] for i in range(0, 68)], dtype=dtype)

    @staticmethod
    def cvface2dlibrects(cvfaces):
        return DlibRectangles([DlibRectangle(*cvface[:2], *(cvface[:2] + cvface[2:]))
                               for cvface in cvfaces])

    def find_faces(self, gray_image, **kwargs):
        rectangles = self.detector(image=gray_image, **kwargs)
        return rectangles

    def estimate_landmarks(self, gray_image, rectangles):

        # transform rectangles data to Dlib-friendly
        dlib_rectangles = self.cvface2dlibrects(rectangles)

        return [self.shape_to_np(self.predictor(gray_image, rectangle)) for rectangle in dlib_rectangles]

    def extract_faces(self, image, **kwargs):

        # transform image to grayscale
        gray = to_grayscale(image)

        # detect faces regions
        rectangles = self.find_faces(gray, **kwargs)
        return rectangles, self.estimate_landmarks(gray, rectangles)

