from numpy import zeros
from numpy import hstack
from numpy import vstack
from numpy import array
from numpy.linalg import inv
from numpy import ndarray

from cv2 import imread
from cv2 import resize
from cv2 import imshow
from cv2 import Rodrigues
from cv2 import projectPoints
from cv2 import VideoCapture
from cv2 import VideoWriter
from cv2 import waitKey
from cv2 import cvtColor
from cv2 import COLOR_GRAY2RGB
from cv2 import COLOR_BGRA2BGR
from cv2 import VideoWriter_fourcc
from cv2 import getTickFrequency
from cv2 import getTickCount
from cv2 import flip
from numpy import newaxis
from pathlib import Path

from pykinect2.PyKinectRuntime import PyKinectRuntime
from pykinect2.PyKinectV2 import FrameSourceTypes_Color
from pykinect2.PyKinectV2 import FrameSourceTypes_Body

from pypylon import factory
from game.estimator import to_grayscale
fourcc = VideoWriter_fourcc(*'MPEG')


def time_measure(func):
    def wrapper(*args, **kwargs):
        e1 = getTickCount()
        res = func(*args, **kwargs)
        e2 = getTickCount()
        print(getTickFrequency() / (e2 - e1))
        return res

    return wrapper


class Device:

    _translation_shape: tuple = (1, 3)
    _rotation_shape: tuple = (1, 3)
    _extrinsic_matrix_shape: tuple = (4, 4)
    _rotation_matrix_shape: tuple = (3, 3)

    _rotation: ndarray = zeros(_rotation_shape, dtype='float')
    _translation: ndarray = zeros(_translation_shape, dtype='float')
    _rotation_matrix: ndarray = zeros(_rotation_matrix_shape, dtype='float')
    _extrinsic_matrix: ndarray = zeros(_extrinsic_matrix_shape, dtype='float')

    _self_normal: ndarray = array([[0, 0, 1]], dtype='float')
    _origin_normal: ndarray = array([[0, 0, 1]], dtype='float')

    _index: int
    _scale: (int, float) = 1

    _devices: dict = {}

    def __init__(self, index=0, name='device', translation=None, rotation=None, scale=None):

        self.index = index
        self.name = name

        self._devices[self.name] = self

        if scale is not None:
            self.scale = scale

        # extrinsic parameters
        if rotation is not None:
            self.rotation = self.to_ndarray(rotation, self._rotation_shape) / self.scale
        if translation is not None:
            self.translation = self.to_ndarray(translation, self._translation_shape) / self.scale

    def __repr__(self):
        return f'{self.name}'

    def __str__(self):
        return f'{self.name}'

    @staticmethod
    def to_ndarray(arr, shape):
        if arr is not None:
            return array(arr).reshape(*shape)
        else:
            return zeros(shape)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        assert isinstance(value, int)
        self._index = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        assert isinstance(value, (int, float))
        self._scale = value

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, value):
        assert isinstance(value, ndarray)
        self._rotation_matrix = value

    @property
    def extrinsic_matrix(self):
        return self._extrinsic_matrix

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, value):
        assert isinstance(value, ndarray)
        self._extrinsic_matrix = value

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, value):
        assert isinstance(value, ndarray)
        self._translation = value
        self.extrinsic_matrix = self.restore_extrinsic_matrix()
        self.normal = self.calculate_normal()

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        assert isinstance(value, ndarray)
        self._rotation = value
        self.rotation_matrix = self.create_rotation_matrix(self.rotation)
        self.normal = self.calculate_normal()

    @staticmethod
    def create_rotation_matrix(rotation):
        """(1, 3) -> (3, 3)"""
        return Rodrigues(rotation)[0]

    def restore_extrinsic_matrix(self):
        """(3, 3), (1, 3), (4,) -> (4, 4)"""
        return vstack((hstack((self.rotation_matrix,
                               self.translation.T)),
                       array([0.0, 0.0, 0.0, 1.0])))

    def calculate_normal(self):
            return self.vectors_to_origin(self._self_normal, translation=False)

    @property
    def normal(self):
        return self._origin_normal

    @normal.setter
    def normal(self, value):
        assert isinstance(value, ndarray)
        self._origin_normal = value

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

        if translation and hasattr(self, 'translation'):
            return (self.rotation_matrix @ vectors.T + self.translation.T).T
        else:
            return (self.rotation_matrix @ vectors.T).T

    @classmethod
    def get(cls, name):
        return cls._devices.get(name)

    @classmethod
    def pop(cls, name):
        cls._devices.pop(name)

    @classmethod
    def clear(cls):
        cls._devices = {}

    @classmethod
    def items(cls):
        return cls._devices.items()

    @classmethod
    def keys(cls):
        return cls._devices.keys()

    @classmethod
    def values(cls):
        return cls._devices.values()


class Picture(Device):

    _picture: ndarray
    _filename: (str, Path)
    _devices: dict = {}

    def __init__(self, index, name, filename=None, **kwargs):
        super().__init__(index=index,
                         name=name,
                         translation=kwargs.get('translation'),
                         rotation=kwargs.get('rotation'),
                         scale=kwargs.get('scale'))
        if filename is not None:
            self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        assert isinstance(value, (str, Path))
        self._filename = value

    @property
    def picture(self):
        return self._picture

    @picture.setter
    def picture(self, value):
        assert isinstance(value, ndarray)
        self._picture = value

    def load_pic(self, img_path, flags=-1, factor=None):
        image = imread(str(img_path / self.filename), flags=flags)
        if factor is None:
            self.picture = image
        else:
            self.picture = resize(image, (0, 0), fx=1/factor, fy=1/factor)
        return self

    def show_pic(self, winname=None):
        winname = winname if winname is not None else self.name
        try:
            imshow(winname, self.picture)
        except AttributeError:
            return False


class Camera(Device):

    _matrix_shape: tuple = (3, 3)
    _distortion_shape: tuple = (4,)

    _devices: dict = {}
    _connected: bool
    _matrix: ndarray = zeros(_matrix_shape, dtype='float')
    _distortion: ndarray = zeros(_distortion_shape, dtype='float')

    def __init__(self, index=0, name='camera', matrix=None, distortion=None, **kwargs):
        super().__init__(index=index,
                         name=name,
                         translation=kwargs.get('translation'),
                         rotation=kwargs.get('rotation'),
                         scale=kwargs.get('scale'))

        self.connected = False
        if matrix is not None:
            self.matrix = self.to_ndarray(matrix, self._matrix_shape)
        if distortion is not None:
            self.distortion = self.to_ndarray(distortion, self._distortion_shape)

    @property
    def connected(self):
        return self._connected

    @connected.setter
    def connected(self, flag):
        self._connected = flag

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        assert isinstance(value, ndarray)
        self._matrix = value

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, value):
        assert isinstance(value, ndarray)
        self._distortion = value

    def project_points(self, points):
        return projectPoints(points,
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

    _shape: tuple = (480, 640, 3)

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.shape = self.read_image_shape()
        self.connected = True

    def create_out(self):
        self.out = VideoWriter(f'D://test_brs_db//{self.name}.avi', fourcc, 25.0, self.resolution)
        return self

    def check(self):
        test_frame = self.get_frame()
        if test_frame is not None:
            return True
        else:
            self.connected = False
            return False

    def get_frame(self):
        return zeros(self.shape, dtype='uint8')

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

    _device_address = 0

    def __init__(self, index=0, name='WebCamera', device_address=None, **kwargs):
        if device_address is not None:
            self.device_address = device_address
        self.start()
        TypeCamera.__init__(self, index=index, name=name, **kwargs)

    @property
    def device_address(self):
        return self._device_address

    @device_address.setter
    def device_address(self, value):
        assert isinstance(value, int)
        self._device_address = value

    def get_frame(self):
        return self.read()[1]

    def start(self):
        VideoCapture.__init__(self, self.device_address)

    def stop(self):
        self.release()


class KinectColor(TypeCamera, PyKinectRuntime):

    _flip_axis = 1

    def __init__(self, name='KinectColor', **kwargs):
        PyKinectRuntime.__init__(self, FrameSourceTypes_Color)
        TypeCamera.__init__(self, name=name, **kwargs)

    def read_image_shape(self):
        return 1080, 1920, 4

    def get_frame(self):
        if self.has_new_color_frame():
            return flip(self.get_last_color_frame().reshape(self.shape), self._flip_axis)
        else:
            return None

    def start(self):
        pass

    def release(self):
        pass


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

    kinect = KinectColor().create_out()
    ir = InfraredCamera().create_out()
    ir.change_properties(ExposureTime=60000, GainAuto='Off', ExposureAuto='Continuous')
    web = WebCamera().create_out()

    cams = [web, ir, kinect]
    for i in range(10000):
    # while not waitKey(1) == 27:
        e1 = getTickCount()
        # frames = [cam.get_frame() for cam in cams]
        for cam in cams:
            frame = cam.get_frame()
            if frame is not None:
                if frame.shape[-1] == 4:
                    cam.out.write(cvtColor(frame, COLOR_BGRA2BGR))
                else:
                    cam.out.write(frame)

        e2 = getTickCount()

        fps = getTickFrequency() / (e2 - e1)
        time = (e2 - e1) / getTickFrequency() * 1000

        if time < 30:
            continue

        # for i, frame in enumerate(frames):
        #     if frame is None:
        #         break
        #     else:
        #         imshow(str(i), resize(frame, (0, 0), fx=0.5, fy=0.5))

    for cam in cams:
        cam.out.release()
