import tensorflow as tf
import keras.backend as K


def dice(targets, inputs, smooth=1e-6):
    axis = [1, 2, 3]
    intersection = K.sum(targets * inputs, axis=axis)
    return (2 * intersection + smooth) / (K.sum(targets, axis=axis) + K.sum(inputs, axis=axis) + smooth)


def bce_loss(targets, inputs, smooth=1e-6):
    axis = [1, 2, 3]
    # the next line of code might display a warning in some editors (about expected types)
    # this warning does not influence the program, and may be safely ignored
    return - K.sum(targets * tf.math.log(inputs + smooth) + (1 - targets) * tf.math.log(1 - inputs + smooth), axis=axis)


def bce_dice_loss(targets, inputs):
    return bce_loss(targets, inputs) - tf.math.log(dice(targets, inputs))
