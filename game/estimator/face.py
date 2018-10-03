from .utils import to_grayscale

from dlib import shape_predictor
from dlib import rectangle as DlibRectangle
from dlib import rectangles as DlibRectangles

from cv2 import resize
from cv2 import CascadeClassifier

from numpy import array


class FaceDetector:

    def __init__(self, path_to_face_points, path_to_hc_model, factor, scale=1.3, minNeighbors=5):

        # load_weigths face detector model
        self.detector = CascadeClassifier(str(path_to_hc_model)).detectMultiScale

        # parameters for face detector model
        self.scale = scale
        self.minNeighbors = minNeighbors
        self.factor = factor

        # load_weigths face landmarks detector model
        self.predictor = shape_predictor(str(path_to_face_points))

    def upscale(self, coordinates):
        return coordinates * self.factor

    def rescale_coordinates(self, coords):
        return (coords * self.factor).astype(int)

    def downscale(self, image, **kwargs):
        return resize(image, tuple(map(lambda ax: int(ax / self.factor), image.shape[::-1])), **kwargs)

    @staticmethod
    def shape_to_np(shape, dtype='int'):
        return array([[shape.part(i).x, shape.part(i).y] for i in range(0, 68)], dtype=dtype)

    @staticmethod
    def cvface2dlibrects(cvfaces):
        return DlibRectangles([DlibRectangle(*cvface[:2], *(cvface[:2] + cvface[2:]))
                               for cvface in cvfaces])

    @staticmethod
    def _vectors_from_model_to_origin(vectors, matrix, translation_vector, camera):
        return camera.vectors_to_origin(matrix @ vectors.reshape(3, -1) + translation_vector)

    def extract_faces(self, image):

        # transform image to grayscale
        gray = to_grayscale(image)

        # detect faces on downscaled image
        # and return cv2-friendly rectangles with upscaled coordinates
        rectangles = self.upscale(self.detector(self.downscale(gray),
                                                scaleFactor=self.scale,
                                                minNeighbors=self.minNeighbors))

        # transform rectangles data to Dlib-friendly
        dlib_rectangles = self.cvface2dlibrects(rectangles)

        # faces = []
        # for rectangle in dlib_rectangles:
        #     dlib_face = self.predictor(gray, rectangle)
        #     face = self.shape_to_np(dlib_face)
        #     faces.append(face)

        # return raw 2d dlib faces landmarks
        # return rectangles, faces
        return rectangles, [self.shape_to_np(self.predictor(gray, rectangle)) for rectangle in dlib_rectangles]
