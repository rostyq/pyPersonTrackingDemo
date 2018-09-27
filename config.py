PATH_TO_ESTIMATOR = 'game/bin/estimator.h5'  # './checkpoints/model_200_0.0028.h5'  # './app/game/bin/estimator.h5'
PATH_TO_FACE_POINTS = 'game/bin/face_landmarks.dat'
PATH_TO_FACE_MODEL = 'game/bin/face_points_tutorial.mat'
PATH_TO_HAARCASCADE_MODEL = 'game/bin/haarcascade_frontalface_default.xml'
PATH_TO_CAM_DATA = 'game/bin/cam_data.json'
PATH_TO_GAZE_MODEL = 'game/bin/gaze_model.h5'

FACE_DETECTOR = {
    'path_to_face_points': PATH_TO_FACE_POINTS,
    'path_to_hc_model': PATH_TO_HAARCASCADE_MODEL,
    'factor': 2,
    'scale': 1.3,
    'minNeighbors': 5,
}

LANDMARKS_HANDLER = {
    'path_to_face_model': PATH_TO_FACE_MODEL,
    'chin_nose_distance': 0.065
}