import os.path
import argparse
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from utils.losses import dice, bce_dice_loss
from utils.model import UNetModel

VALIDATION_LENGTH = 2000
TEST_LENGTH = 2000
BATCH_SIZE = 32
BUFFER_SIZE = 100
IMG_SHAPE = (256, 256)
NUM_CLASSES = 2


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs for training')
    return parser.parse_args()


def rle_to_mask(rle: str, shape=(768, 768)):
    """
    :param rle: run length encoded pixels as string formatted
    :param shape: (height,width) of array to return
    :return: numpy 2D array, 1 - mask, 0 - background
    """
    encoded_pixels = np.array(rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


def main():
    args = parse_args()

    # set the random seed for reproducibility:
    RANDOM_SEED = 77
    random.seed(RANDOM_SEED)

    TRAIN_DIR = './training_images/'

    df = pd.read_csv("./train_ship_segmentations_v2.csv")
    df['EncodedPixels'] = df['EncodedPixels'].astype('string')

    # Delete corrupted images
    CORRUPTED_IMAGES = ['6384c3e78.jpg']
    df = df.drop(df[df['ImageId'].isin(CORRUPTED_IMAGES)].index)

    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    # Dataframe that contains the segmentation of all ships in the image.
    image_segmentation = shuffled_df.groupby(by=['ImageId'])['EncodedPixels'].apply(
        lambda x: np.nan if pd.isna(x).any() else ' '.join(x)).reset_index()

    print(f'Image segmentation: {image_segmentation.head()}')

    def load_train_image(tensor) -> tuple:
        path = tf.get_static_value(tensor).decode("utf-8")

        image_id = os.path.basename(path)
        input_image = cv2.imread(path)
        input_image = tf.image.resize(input_image, IMG_SHAPE)
        input_image = tf.cast(input_image, tf.float32) / 255.0

        encoded_mask = image_segmentation[image_segmentation['ImageId'] == image_id].iloc[0]['EncodedPixels']
        input_mask = np.zeros(IMG_SHAPE + (1,), dtype=np.int8)
        if not pd.isna(encoded_mask):
            input_mask = rle_to_mask(encoded_mask)
            input_mask = cv2.resize(input_mask, IMG_SHAPE, interpolation=cv2.INTER_AREA)
            input_mask = np.expand_dims(input_mask, axis=2)
        one_hot_segmentation_mask = one_hot(input_mask, NUM_CLASSES)
        input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)

        class_weights = tf.constant([0.0005, 0.9995], tf.float32)
        sample_weights = tf.gather(class_weights, indices=tf.cast(input_mask_tensor, tf.int32),
                                   name='cast_sample_weights')

        return input_image, input_mask_tensor, sample_weights

    images_without_ships = df['EncodedPixels'].isna().sum()
    print(f'df has {len(df) - images_without_ships} images with ships.')

    IMAGES_WITHOUT_SHIPS_NUMBER = 3500

    # reduce the number of images without ships
    images_without_ships = image_segmentation[image_segmentation['EncodedPixels'].isna()]['ImageId'].values[
                           :IMAGES_WITHOUT_SHIPS_NUMBER]
    images_with_ships = image_segmentation[image_segmentation['EncodedPixels'].notna()]['ImageId'].values[:6000]
    images_list = np.append(images_without_ships, images_with_ships)

    # remove corrupted images
    images_list = np.array(list(filter(lambda x: x not in CORRUPTED_IMAGES, images_list)))

    print(f'Length of image list: {len(images_list)}')

    TRAIN_LENGTH = len(images_list) - VALIDATION_LENGTH - TEST_LENGTH

    # Shuffle the rows
    np.random.shuffle(images_list)

    print(f'Number of training images: {TRAIN_LENGTH}')

    images_list = tf.data.Dataset.list_files([f'{TRAIN_DIR}{name}' for name in images_list], shuffle=True)
    train_images = images_list.map(lambda x: tf.py_function(load_train_image, [x], [tf.float32, tf.float32]),
                                   num_parallel_calls=tf.data.AUTOTUNE)
    train_images = train_images.map(
        lambda img, mask: (tf.ensure_shape(img, IMG_SHAPE + (3,)), tf.ensure_shape(mask, IMG_SHAPE + (NUM_CLASSES,))),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    validation_dataset = train_images.take(VALIDATION_LENGTH)
    test_dataset = train_images.skip(VALIDATION_LENGTH).take(TEST_LENGTH)
    train_dataset = train_images.skip(VALIDATION_LENGTH + TEST_LENGTH)

    train_batches = (
        train_dataset
        .repeat()
        .batch(BATCH_SIZE))
    print('Train batches:')
    print(train_batches.take(1))

    validation_batches = validation_dataset.batch(BATCH_SIZE)

    test_batches = test_dataset.batch(BATCH_SIZE)

    EPOCHS = args.n_epochs
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    optimizer = tfa.optimizers.RectifiedAdam(
        learning_rate=0.01,
        total_steps=EPOCHS * STEPS_PER_EPOCH,
        warmup_proportion=0.2,
        min_lr=0.00001,
    )
    optimizer = tfa.optimizers.Lookahead(optimizer)

    model = UNetModel(IMG_SHAPE + (3,)).model
    model.compile(optimizer=optimizer,
                  loss=bce_dice_loss,
                  metrics=[dice], )

    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print(f'Trainable params: {trainable_params}')

    print(model.summary())

    model_history = model.fit(train_batches,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_data=validation_batches)

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'C2', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

    dices = model_history.history['dice']
    val_dices = model_history.history['val_dice']

    plt.figure()
    plt.plot(model_history.epoch, dices, 'm', label='Training Dice')
    plt.plot(model_history.epoch, val_dices, 'y', label='Validation Dice')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()

    results = model.evaluate(test_batches)
    print("test loss, test dice:", results)

    # Specify the file path to save the model
    hdf5_path = './models/model.h5'

    # Save the model as an HDF5 file
    model.save(hdf5_path, save_format='h5')


if __name__ == "__main__":
    main()