# from __future__ import absolute_import, division, print_function, unicode_literals
import zipfile
import os
import logging

## TODO: move to tflite
## TODO: move away from IBM art for FGSM
## Optimize and deploy at edge
import time

# import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Input
import keras.backend as k
from keras.preprocessing import image
import numpy as np
import cv2 as cv
## Disable eager execution for compatibility with IBM art
## TODO: remove this and migrate away from art
tf.compat.v1.disable_eager_execution()

from mtcnn import MTCNN

mtdetector = MTCNN()

from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod
from deepface import create_deepface, get_weights

## pre load weights on app startup so that we aren't pulling on every request
weights = get_weights()

def load_deepface():
    # from DeepFace import create_deepface, get_weights
    model = create_deepface()
    model.load_weights(weights)
    return model

## Creates a mask for the image, extracts out facial feature predictions starting
## from `facial_feature_index`
def create_feature_mask_mtcnn(
    detections, img_shape, mask_pars=(0.03, 0.05, 0.03, 0.06, 0.04, 0.03)
):
    ## mask_pars=(eye_a, eye_b,  mouth_a, nose_a, nose_b, nose_y) is a tuple of hyperparameters.
    ## par_a and par_b are the prinicpal axes lengths in the y- and x-directions
    ## nose_y is the offset of the nose ellipse center in the y-direction
    mask = np.ones(shape=img_shape)
    eye_a, eye_b, mouth_a, nose_a, nose_b, nose_y = [
        x * img_shape[0] for x in mask_pars
    ]

    for i in range(len(detections)):
        keypoints = detections[i]["keypoints"]
        mouth_b = (keypoints["mouth_right"][0] - keypoints["mouth_left"][0]) / 2
        for feat, pt in detections[i]["keypoints"].items():
            if feat != "mouth_left":
                kp_x, kp_y = pt

                mask_x, mask_y = np.ogrid[
                    -kp_y : img_shape[0] - kp_y, -kp_x : img_shape[1] - kp_x
                ]
                if feat == "left_eye" or feat == "right_eye":
                    kernel_mask = (
                        mask_x ** 2 / eye_a ** 2 + mask_y ** 2 / eye_b ** 2 <= 1
                    )
                if feat == "mouth_right":
                    kernel_mask = (
                        mask_x ** 2 / mouth_a ** 2
                        + (mask_y + mouth_b) ** 2 / mouth_b ** 2
                        <= 1
                    )
                if feat == "nose":
                    kernel_mask = (
                        mask_x + nose_y
                    ) ** 2 / nose_a ** 2 + mask_y ** 2 / nose_b ** 2 <= 1

                mask[kernel_mask] = 8e-1

    return mask


## Function to apply mask to an adversarial image.
## Takes a hyper parameter of the mask width
def apply_mask_to_adv_noise_mtcnn(original_image, adv_image):
    ## Use facial feature extraction
    frame = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, (128, 128))
    prediction = mtdetector.detect_faces(frame)

    ## Get mask
    mask = create_feature_mask_mtcnn(prediction, frame.shape)
    mask = mask.astype(np.float32)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

    ## calculate the diff between the adversarial image and the original image
    noise = adv_image - original_image.astype(np.float32)

    noise = cv.resize(noise, (128, 128))

    ## Limitation of blazeface is that it needs us to downsample the original image
    ## A bit more tweaking can work for higher resolution images
    original_resized_image = cv.resize(original_image, (128, 128))

    ## Apply mask and return
    return (
        np.where(
            mask < 1, noise + original_resized_image, original_resized_image
        ).astype(np.uint),
        noise,
    )  ## send back noise for debugging


def create_classifier(model):
    ## Specify image manipulation masks
    ## These are fed into the classifier __init__ to modify new data points fed into the network
    mean_imagenet = np.zeros([152, 152, 3])
    mean_imagenet[..., 0].fill(103.939)
    mean_imagenet[..., 1].fill(116.779)
    mean_imagenet[..., 2].fill(123.68)
    classifier = KerasClassifier(
        clip_values=(0, 255), model=model, preprocessing=(mean_imagenet, 1)
    )
    return classifier


## Let's run this through the FastGradientMethod Attack


def run_fgsm_attacks(
    classifier,
    target_image,
    eps,
    mask_width=20,
    masked=True,
    img_show=True,
    debug=True,
    use_art=True,
    feature_extractor="blazeface",
    iter_step=1,
):
    adv_image = None
    if use_art:
        attack = FastGradientMethod(classifier=classifier, eps=eps)
        x_adv = None
        for i in range(iter_step):
            try:
                start = time.time()
                x_adv = attack.generate(
                    x=np.array([target_image]), x_adv_init=x_adv, resume=True
                )
                end = time.time()

                ### Apply mask
                adv_image = x_adv[0].astype(np.uint)
                target_image_copy = target_image.copy()
                if masked:
                    if feature_extractor == "blazeface":
                        adv_image, _ = apply_mask_to_adv_noise(
                            target_image, adv_image, mask_width=mask_width
                        )
                    else:
                        adv_image, _ = apply_mask_to_adv_noise_mtcnn(
                            target_image, adv_image
                        )
                    target_image_copy = cv.resize(target_image_copy, (128, 128))

                norm = np.linalg.norm(
                    np.reshape(adv_image - target_image_copy, [-1]), ord=np.inf
                )
                logging.debug(f'debug: norm: {norm}')

            except Exception as e:
                logging.error(e)
            attack.max_iter = iter_step
    else:
        raise NotImplementedError
        # loss_object = keras.losses.CategoricalCrossentropy()
        # with tf.GradientTape() as tape:
        #     tape.watch(target_image)
        #     prediction = classifier(target_image)
        #     loss = loss_object(prediciton, )
    return adv_image


def generate_adv_masked_image(image_path):
    target_image = image.img_to_array(
        image.load_img(image_path, target_size=(152, 152))
    )

    deepface_classifier = create_classifier(load_deepface())

    adv_image = run_fgsm_attacks(
        deepface_classifier,
        target_image,
        eps=13,
        mask_width=30,
        masked=True,
        feature_extractor="mtcnn",
    )
    return adv_image

import argparse

if __name__ == "__main__":
    logging.info("loaded patchface")
    parser = argparse.ArgumentParser(description='Create Masked Adversarial Attacks')

    parser.add_argument('--image-path', metavar='image_path', required=True, help='Path to image file to add mask to')
    parser.add_argument('--save-path', metavar='save_path', required=False, help='Path to image file to add mask to', default='./test.jpeg')
    args = parser.parse_args()

    adv_image = generate_adv_masked_image(args.image_path)

    logging.debug(adv_image)

    cv.imwrite('/tmp/file.jpg', adv_image)
    
    cv.imwrite(args.save_path, cv.cvtColor(cv.imread('/tmp/file.jpg'), cv.COLOR_BGR2RGB))