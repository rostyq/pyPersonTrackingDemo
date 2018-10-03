from game.dev.devices import Picture
from game.dev.devices import Camera
from game.dev.devices import Device
from game.dev.devices import KinectColor
from game.dev.devices import InfraredCamera
from game.dev.devices import WebCamera
from pathlib import Path
from json import load

KinectColorName = 'KinectColor'
WebCameraName = 'WebCamera'
InfraredCameraName = 'InfraredCamera'
KinectInfraredName = 'KinectInfrared'

indentification_keys = ['index', 'class_type', 'name']


camera_types = {
    KinectColorName: KinectColor,
    InfraredCameraName: InfraredCamera,
    WebCameraName: WebCamera
}


def get_identification(data: dict):
    return iter(data.pop(key) for key in indentification_keys)


def load_devices(database_file, img_path=None, scale=1, factor=4):

    # load json file with devices parameters
    with open(database_file, 'r') as f:
        devices = load(f)

    for device in devices:
        index, class_type, name = get_identification(device)
        if class_type == 'Camera':
            camera_types.get(name, Camera)(index=index, name=name, scale=scale, **device)
        elif class_type == 'Picture':
            Picture(index=index, name=name, scale=scale, **device).load_pic(img_path, factor=factor)
        else:
            Device(index=index, name=name, scale=scale, **device)


if __name__ == '__main__':
    from cv2 import waitKey
    from config import BIN_PATH
    from config import DATABASE_FILE
    from config import IMAGE_DIR

    root_dir = Path('../../')
    img_path = root_dir / BIN_PATH / IMAGE_DIR
    database_file = root_dir / BIN_PATH / DATABASE_FILE

    print(img_path)
    load_devices(database_file, img_path)

    for pic in Picture.values():
        print(pic, type(pic))
        pic.show_pic()
        while not waitKey(1) == 27:
            pass