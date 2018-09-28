from game import GameRuntime

from config import FACE_DETECTOR
from config import PATH_TO_CAM_DATA
from config import LANDMARKS_HANDLER
from config import PATH_TO_GAZE_MODEL

from numpy import set_printoptions

set_printoptions(formatter={'float': '{: 0.3f}'.format})

__main__ = "Kinect v2 Body Game"


def main():

    game = GameRuntime(FACE_DETECTOR, LANDMARKS_HANDLER, PATH_TO_GAZE_MODEL, PATH_TO_CAM_DATA)
    game.run()


if __name__ == '__main__':
    main()
