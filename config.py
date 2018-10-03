from pathlib import Path

BIN_PATH = Path('game/bin/')
IMAGE_DIR = 'img'
DATABASE_FILE = 'db.json'
FACE_POINTS_FILE = 'face_landmarks.dat'
FACE_MODEL_FILE = 'face_points_tutorial.mat'
HAARCASCADE_MODEL_FILE = 'haarcascade_frontalface_default.xml'
GAZE_MODEL_FILE = 'gaze_model.h5'

FACE_DETECTOR = {
    'path_to_face_points': BIN_PATH / FACE_POINTS_FILE,
    'path_to_hc_model': BIN_PATH / HAARCASCADE_MODEL_FILE,
    'factor': 2,
    'scale': 1.3,
    'minNeighbors': 5,
}

LANDMARKS_HANDLER = {
    'path_to_face_model': BIN_PATH / FACE_MODEL_FILE,
    'chin_nose_distance': 0.065
}

LOAD_DATA = {
    'database_file': BIN_PATH / DATABASE_FILE,
    'img_path': BIN_PATH / IMAGE_DIR,
    'scale': 1,
    'factor': 4
}
