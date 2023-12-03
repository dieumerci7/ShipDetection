import argparse
import tensorflow as tf
import cv2
import numpy as np
from utils.losses import bce_dice_loss, dice
import tensorflow_addons as tfa  # important, otherwise we get a custom_object_scope-related error


def parse_args():
    parser = argparse.ArgumentParser(description="Image Segmentation Inference")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_name", type=str, required=True, help="Name of the output image")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    loaded_model = tf.keras.models.load_model('./models/final_model.h5',
                                              custom_objects={'bce_dice_loss': bce_dice_loss, 'dice': dice})

    image = cv2.imread(args.input_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    pred_mask = loaded_model.predict(image)[0].argmax(axis=-1)
    mask_resized = cv2.resize(pred_mask, (768, 768), interpolation=cv2.INTER_NEAREST)

    # Convert to uint8
    mask_resized_uint8 = (mask_resized * 255).astype('uint8')

    output_name = args.output_name
    # Save the mask as a JPEG file
    cv2.imwrite(f'./segmented_images/{output_name}', mask_resized_uint8)
    print(f"Segmented image saved at: {output_name}")

    # Create a colormap with black and white
    colored_mask = cv2.cvtColor(mask_resized_uint8, cv2.COLOR_GRAY2BGR)

    # Display the colored mask using OpenCV
    cv2.imshow('Resized Mask', colored_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()