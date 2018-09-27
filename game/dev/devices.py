from numpy import zeros
from numpy import hstack
from numpy import vstack
from numpy import array
from numpy.linalg import inv

from cv2 import Rodrigues
from cv2 import projectPoints
from cv2 import VideoCapture
from cv2 import resize
from cv2 import imshow
from cv2 import waitKey
from cv2 import cvtColor
from cv2 import COLOR_GRAY2RGB

from cv2 import getTickFrequency
from cv2 import getTickCount

from pykinect2.PyKinectRuntime import PyKinectRuntime
from pykinect2.PyKinectV2 import FrameSourceTypes_Color
from pykinect2.PyKinectV2 import FrameSourceTypes_Body

from pypylon import factory

# _kinect = PyKinectRuntime(FrameSourceTypes_Color)

def time_measure(func):
    def wrapper(*args, **kwargs):
        e1 = getTickCount()
        res = func(*args, **kwargs)
        e2 = getTickCount()
        print(getTickFrequency() / (e2 - e1))
        return res

    return wrapper


class Device:

    default_scale = 1
    default_translation = zeros((1, 3), dtype='float')
    default_rotation = zeros((1, 3), dtype='float')

    devs = {}

    def __init__(self, name='device', translation=None, rotation=None, scale=None):

        self.name = name

        self.devs[self.name] = self

        self.scale = scale if scale else self.default_scale

        # extrinsic parameters
        self._rotation = array(rotation).reshape(1, 3) / self.scale if rotation else self.default_rotation
        self._translation = array(translation).reshape(1, 3) / self.scale if translation else self.default_translation

        # create and update matrices
        self.rotation_matrix = None
        self.extrinsic_matrix = None
        self.update_rotation_matrix()
        self.update_extrinsic_matrix()

    @property
    def translation(self):
        self._translation = None

    @translation.getter
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        self._translation = array(value).reshape(1, 3)
        self.update_extrinsic_matrix()

    @property
    def rotation(self):
        self._rotation = None
        self.rotation_matrix = None

    @rotation.getter
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value.reshape(1, 3)
        self.update_rotation_matrix()
        self.update_extrinsic_matrix()

    @staticmethod
    def create_rotation_matrix(rotation):
        """(1, 3) -> (3, 3)"""
        return Rodrigues(rotation)[0]

    @staticmethod
    def restore_extrinsic_matrix(rotation_matrix, translation):
        """(3, 3), (1, 3), (4,) -> (4, 4)"""
        return vstack((hstack((rotation_matrix,
                               translation.T)),
                       array([0.0, 0.0, 0.0, 1.0])))

    def update_rotation_matrix(self):
        self.rotation_matrix = self.create_rotation_matrix(self.rotation)

    def update_extrinsic_matrix(self):
        self.extrinsic_matrix = self.restore_extrinsic_matrix(self.rotation_matrix,
                                                              self.translation)

    def vectors_to_self(self, vectors, translation=True):
        """
        (?, 3) -> (?, 3)

        (inv((3, 3)) @ ( (?, 3) - (1, 3) ).T).T -> (?, 3)
        """
        assert vectors.ndim == 2
        assert vectors.shape[1] == 3

        if translation:
            return (inv(self.rotation_matrix) @ (vectors - self.translation).T).T
        else:
            return (inv(self.rotation_matrix) @ vectors.T).T

    def vectors_to_origin(self, vectors, translation=True):
        """
        (?, 3) -> (?, 3)

        (inv((3, 3)) @ ( (?, 3) - (1, 3) ).T).T -> (?, 3)
        """
        assert vectors.ndim == 2
        assert vectors.shape[1] == 3

        if translation:
            return (self.rotation_matrix @ vectors.T + self.translation.T).T
        else:
            return (self.rotation_matrix @ vectors.T).T

    @classmethod
    def get(cls, name):
        return cls.devs.get(name)

    @classmethod
    def pop(cls, name):
        cls.devs.pop(name)

    @classmethod
    def clear(cls):
        cls.devs = {}

    @classmethod
    def items(cls):
        return cls.devs.items()

    @classmethod
    def keys(cls):
        return cls.devs.keys()

    @classmethod
    def values(cls):
        return cls.devs.values()


class Camera(Device):

    _connected: bool
    default_matrix = zeros((3, 3), dtype='float')
    default_distortion = zeros((4,), dtype='float')

    def __init__(self, name='camera', matrix=None, distortion=None, **kwargs):
        translation = kwargs.get('translation')
        rotation = kwargs.get('rotation')
        scale = kwargs.get('scale')
        super().__init__(name, translation=translation, rotation=rotation, scale=scale)

        self.connected = False
        self.matrix = array(matrix) if matrix else self.default_matrix
        self.distortion = array(distortion) if distortion else self.default_distortion

    @property
    def connected(self):
        return self._connected

    @connected.setter
    def connected(self, flag):
        self._connected = flag

    def project_vectors(self, vectors):
        return projectPoints(vectors,
                             -self.rotation,
                             -(inv(self.rotation_matrix) @ self.translation.T),
                             self.matrix,
                             self.distortion)[0].reshape(-1, 2)

    def find_ray_point(self, image_points, origin=True):
        """
        ((3, 3) @ (?, 3).T).T -> (?, 3)
        """

        assert image_points.ndim == 2
        assert image_points.shape[1] == 3

        ray_points = (inv(self.matrix) @ image_points.T).T
        return self.vectors_to_origin(ray_points) if origin else ray_points


class TypeCamera(Camera):

    _shape: tuple

    default_image_shape = (480, 640, 3)

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = self.read_image_shape()
        self.connected = True
        self.check()

    def check(self):
        self.get_frame()

    def get_frame(self):
        return zeros(self.default_image_shape, dtype='uint8')

    def read_image_shape(self):
        shape = self.get_frame().shape
        return shape if len(shape) == 3 else (*shape, 1)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        assert isinstance(shape, tuple)
        self._shape = shape

    @property
    def resolution(self):
        return self.shape[1::-1]

    @property
    def channels(self):
        return self.shape[-1]

    @property
    def width(self):
        return self.resolution[0]

    @property
    def height(self):
        return self.resolution[1]

    def start(self):
        pass

    def stop(self):
        pass

    def restart(self):
        pass


class WebCamera(TypeCamera, VideoCapture):

    default_device_index = 0

    def __init__(self, name='WebCamera', index=None, **kwargs):
        self.index = index if index is not None else self.default_device_index
        self.start()
        TypeCamera.__init__(self, name=name, **kwargs)

    def get_frame(self):
        return self.read()[1]

    def start(self):
        VideoCapture.__init__(self, self.index)

    def stop(self):
        self.release()


class KinectColor(TypeCamera, PyKinectRuntime):

    def __init__(self, name='KinectColor', **kwargs):
        PyKinectRuntime.__init__(self, FrameSourceTypes_Color)
        TypeCamera.__init__(self, name=name, **kwargs)

    def read_image_shape(self):
        return 1080, 1920, 4

    def get_frame(self):
        if self.has_new_color_frame():
            return self.get_last_color_frame().reshape(self.shape)
        else:
            return None

    def start(self):
        pass

    def release(self):
        pass


# class PylonWrapper:
#
#     default_index = 0
#
#     def __init__(self, index=None):
#         self.index = index if index is not None else self.default_index
#         self._runtime = factory.create_device(factory.find_devices()[self.index])
#
#     def __call__(self, *args, **kwargs):
#         return self._runtime


class InfraredCamera(TypeCamera):

    _frames_factory: iter

    def __init__(self, name='InfraredCamera', **kwargs):
        self._runtime = factory.create_device(factory.find_devices()[0])
        self.start()
        super().__init__(name=name, **kwargs)

    def change_properties(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self._runtime.properties[key] = value
            except OSError:
                print(f'{key} is not writable.')
            except KeyError:
                print(f'{key} not found.')

    def show_properties(self):
        for key in self._runtime.properties.keys():
            try:
                print(key, self._runtime.properties[key])
            except OSError:
                print(f'{key} is not readable.')

    def start(self):
        self._runtime.open()
        self._frames_factory = self._runtime.grab_images(-1)

    def stop(self):
        self._runtime.close()

    def restart(self):
        self.stop()
        self.start()

    # @time_measure
    def get_frame(self):
        try:
            return cvtColor(next(self._frames_factory), COLOR_GRAY2RGB)
        except RuntimeError:
            # self.restart()
            return None
        except StopIteration:
            self.restart()
            return None


if __name__ == '__main__':

    kinect = KinectColor()
    ir = InfraredCamera()
    ir.change_properties(ExposureTime=60000, GainAuto='Off', ExposureAuto='Continuous')
    web = WebCamera()

    cams = [web, ir, kinect]

    while not waitKey(1) == 27:
        e1 = getTickCount()
        frames = [cam.get_frame() for cam in cams]
        e2 = getTickCount()

        fps = getTickFrequency() / (e2 - e1)
        time = (e2 - e1) / getTickFrequency() * 1000

        if time < 30:
            continue

        for i, frame in enumerate(frames):
            if frame is None:
                break
            else:
                imshow(str(i), resize(frame, (0, 0), fx=0.5, fy=0.5))