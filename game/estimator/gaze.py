from game.estimator.utils import vecs2angles
from game.estimator.utils import angles2vecs

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Dropout

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.normalization import BatchNormalization

from keras.initializers import RandomNormal
from keras.initializers import glorot_uniform

from keras.regularizers import l2

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TerminateOnNaN

from keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
from numpy import pi

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from os import path as Path

debug = False
if debug:
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)
else:
    pass


def calc_angle(angles1, angles2):

    def to_vector(angle):
        x = (-1) * tf.cos(angle[:, 1]) * tf.sin(angle[:, 0])
        y = (-1) * tf.sin(angle[:, 1])
        z = (-1) * tf.cos(angle[:, 1]) * tf.cos(angle[:, 0])
        return tf.stack((x, y, z), axis=1)

    def unit_vector(array):
        return tf.divide(array, tf.norm(array, axis=1, keep_dims=True))

    unit_v1, unit_v2 = unit_vector(to_vector(angles1)), unit_vector(to_vector(angles2))

    return tf.acos(
        tf.clip_by_value(tf.reduce_sum(unit_v1 * unit_v2, axis=1), -1.0, 1.0),
        name='acos'
        ) * 180 / pi


def angle_accuracy(target, predicted):
    return tf.reduce_mean(calc_angle(predicted, target), name='mean_angle')


def custom_loss(y_true_and_weights, y_pred):
   y_true, y_weights = y_true_and_weights[:, :-1], y_true_and_weights[:, -1:]
   loss = K.reshape(K.sum(K.square(y_pred - y_true), axis=1), (1, -1))
   return K.dot(2 * pi / y_weights, loss)/K.cast(K.shape(y_true)[0], 'float32')


def create_model(learning_rate=0.001, seed=None, convolution_trainable=True, dropout=0.3, alpha=1e-5):

    # input
    input_img = Input(shape=(72, 120, 1), name='InputImage')
    input_pose = Input(shape=(2,), name='InputPose')

    # regularization
    reg = l2(alpha)

    # convolutional
    conv1 = Conv2D(
        filters=32,
        activation='elu',
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.1, seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=reg,
        name='conv1',
        trainable=convolution_trainable
        )(input_img)
    pool1 = MaxPool2D(
        pool_size=(4, 4),
        strides=(4, 4),
        padding='valid',
        name='maxpool1'
        )(conv1)
    conv2 = Conv2D(
        filters=64,
        activation='elu',
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.1, seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=reg,
        name='conv2',
        trainable=convolution_trainable
        )(pool1)
    pool2 = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        name='maxpool2'
        )(conv2)
    conv3 = Conv2D(
        filters=96,
        activation='elu',
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=reg,
        name='conv3',
        trainable=convolution_trainable
        )(pool2)
    pool3 = MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid',
        name='maxpool3'
        )(conv3)

    flatt = Flatten(name='flatt')(pool3)

    # concatanate with head pose
    cat = Concatenate(axis=-1, name='concat')([flatt, input_pose])

    batch_norm = BatchNormalization()(cat)

    # inner product 1
    dense1 = Dense(
        units=100,
        activation='elu',
        kernel_initializer=glorot_uniform(seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=None,
        name='fc1',
        trainable=convolution_trainable
        )(batch_norm)

    # drop = Dropout(0.3)(dense1)

    dense2 = Dense(
        units=50,
        activation='elu',
        kernel_initializer=glorot_uniform(seed=seed),
        bias_initializer='zeros',
        kernel_regularizer=None,
        name='fc2',
        trainable=convolution_trainable
    )(dense1)

    drop = Dropout(dropout)(dense2)

    # inner product 2
    dense3 = Dense(
        units=2,
        activation='linear',
        kernel_initializer=glorot_uniform(seed=seed),
        bias_initializer='zeros',
        name='fc3',
        trainable=True
        )(drop)

    ### OPTIMIZER ###
    optimizer = SGD(
        lr=learning_rate,
        # decay=0.01,
        nesterov=True,
        momentum=0.9
        )

    ### COMPILE MODEL ###
    model = Model([input_img, input_pose], dense3)
    model.compile(optimizer=optimizer, loss='mse', metrics=[angle_accuracy])
    print(model.summary())
    return model


def create_callbacks(model_name, path_to_save, save_period=100):

    if model_name is None:
        model_name = 'model_{epoch}_{loss:.4f}.h5'
    ### CALLBACKS ###
    tbCallBack = TensorBoard(
        log_dir=path_to_save,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_grads=False
        )
    checkpoint = ModelCheckpoint(
        Path.join(path_to_save, model_name),
        monitor='loss',
        period=save_period
        )
    terminate = TerminateOnNaN()

    return [tbCallBack, checkpoint, terminate]


class GazeNet:

    default_params = {}

    def __init__(self, params=None):
        self.params = params if params is not None else self.default_params
        self.model = create_model(**self.params)
        self.image_shape = self.model.layers[0].get_output_at(0).get_shape()[1:3]

    def load_weigths(self, path_to_model):
        self.model.load_weights(path_to_model)
        return self

    def estimate(self, eye_images, face_gazes):

        # check images
        assert eye_images.ndim == 4
        assert eye_images.shape[-1] == 1
        assert eye_images.shape[1:3] == self.image_shape

        # check face gazes
        assert face_gazes.ndim == 2
        assert face_gazes.shape[-1] == 3

        # transform
        angle_face_gazes = vecs2angles(face_gazes)
        eye_images = eye_images / 255

        return angles2vecs(self.model.predict([eye_images, angle_face_gazes]))


if __name__ == '__main__':

    _gazenet = GazeNet().load_weigths('../../game/bin/gaze_model.h5')
    print(_gazenet.image_shape)
    for layer in _gazenet.model.layers:
        print(layer.get_output_at(0).get_shape())
