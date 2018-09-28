from game.dev.devices import Camera
from game.dev.devices import Device
from game.dev.devices import KinectColor
from game.dev.devices import InfraredCamera
from game.dev.devices import WebCamera

from json import load
from pprint import pprint

defaultKinectColorName = 'KinectColor'
defaultWebCameraName = 'WebCamera'
defaultInfraredCameraName = 'InfraredCamera'
defaultKinectInfraredName = 'KinectInfrared'

cameraTypes = {
    defaultKinectColorName: KinectColor,
    defaultInfraredCameraName: InfraredCamera,
    defaultWebCameraName: WebCamera
}


def load_devices(cam_data_path, scale=1000):

    # load camera parameters
    with open(cam_data_path, 'r') as f:
        cam_data = load(f)

    # read json and create objects
    for cam_name, data_dict in cam_data.items():

        camera_class = cameraTypes.get(cam_name, Camera)
        try:
            cam = camera_class(cam_name, **data_dict, scale=scale)
        except Exception as e:
            print(e)
            cam = Camera(cam_name, **data_dict, scale=scale)

        data_dict.update({'connected': cam.connected})

        print(f'Added camera `{cam}` with parameters:')
        pprint(data_dict, indent=3)
        print()


if __name__ == '__main__':

    from config import PATH_TO_CAM_DATA

    load_devices('../../'+PATH_TO_CAM_DATA)
