# from __future__ import absolute_import, division, print_function, unicode_literals
import zipfile
import os
# import numpy as np
from tensorflow import keras
from keras.layers import Dense, Flatten, Input
import keras.backend as k

from mtcnn import MTCNN

from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod
from deepface import create_deepface, get_weights

def load_deepface():
  # from DeepFace import create_deepface, get_weights
  model = create_deepface()

  weights = get_weights()
  model.load_weights(weights)
  return model

## Creates a mask for the image, extracts out facial feature predictions starting
## from `facial_feature_index`
def create_feature_mask_mtcnn(detections, img_shape, mask_pars=(0.03,0.05,0.03,0.06,0.04,0.03)):
  ## mask_pars=(eye_a, eye_b,  mouth_a, nose_a, nose_b, nose_y) is a tuple of hyperparameters.
  ## par_a and par_b are the prinicpal axes lengths in the y- and x-directions
  ## nose_y is the offset of the nose ellipse center in the y-direction
  mask = np.ones(shape=img_shape)
  eye_a, eye_b, mouth_a, nose_a, nose_b, nose_y = [x * img_shape[0] for x in mask_pars]
    
  for i in range(len(detections)):
    keypoints = detections[i]['keypoints']
    mouth_b = (keypoints['mouth_right'][0] - keypoints['mouth_left'][0]) / 2
    for feat, pt in detections[i]['keypoints'].items():
      if feat != 'mouth_left':
        kp_x, kp_y = pt
        
        mask_x, mask_y = np.ogrid[-kp_y:img_shape[0]-kp_y, -kp_x:img_shape[1]-kp_x]
        if feat == 'left_eye' or feat == 'right_eye':
          kernel_mask = mask_x**2 / eye_a**2 + mask_y**2 / eye_b**2 <= 1
        if feat == 'mouth_right':
          kernel_mask = mask_x**2 / mouth_a**2 + (mask_y + mouth_b)**2 / mouth_b**2 <= 1
        if feat == 'nose':
          kernel_mask = (mask_x + nose_y)**2 / nose_a**2 + mask_y**2 / nose_b**2 <= 1
 #       kernel_mask = mask_x**2 + mask_y**2 <= mask_width ## Draw circle
        

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
  return np.where(mask < 1, noise + original_resized_image, original_resized_image).astype(np.uint), noise ## send back noise for debugging

def create_classifier(model):
  ## Specify image manipulation masks
  ## These are fed into the classifier __init__ to modify new data points fed into the network
  mean_imagenet = np.zeros([152, 152, 3])
  mean_imagenet[...,0].fill(103.939)
  mean_imagenet[...,1].fill(116.779)
  mean_imagenet[...,2].fill(123.68)
  classifier = KerasClassifier(clip_values=(0, 255), model=model, preprocessing=(mean_imagenet, 1))
  return classifier

## Let's run this through the FastGradientMethod Attack

def run_fgsm_attacks(
    classifier, 
    target_image, 
    eps_range=list(range(50, 150, 20)), 
    mask_width=20, 
    masked=True, 
    history=None, ## Track changes and metrics
    ground_label='hugh jackman',
    img_show = True,
    debug=True,
    feature_extractor='blazeface',
    iter_step = 5):
  
  if history is None:
    history = dict()

  for eps in eps_range:
    attack = FastGradientMethod(classifier=classifier, eps=eps)
    x_adv = None
    for i in range(iter_step):
        try:

          start = time.time()
          x_adv = attack.generate(x=np.array([target_image]), x_adv_init=x_adv, resume=True)
          end = time.time()
          
          ### Apply mask
          adv_image = x_adv[0].astype(np.uint)
          target_image_copy = target_image.copy()
          if masked:
            if feature_extractor == 'blazeface':
              adv_image, _ = apply_mask_to_adv_noise(target_image, adv_image, mask_width=mask_width)
            else:
              adv_image, _ = apply_mask_to_adv_noise_mtcnn(target_image, adv_image)
            target_image_copy = cv.resize(target_image_copy, (128, 128))

          norm = np.linalg.norm(np.reshape(adv_image - target_image_copy, [-1]), ord=np.inf)

          if norm not in history.keys():
            history[norm] = 0 ## Store correct predictions

          if debug:
            print(
                f"Adversarial image at step {(i * iter_step)} \n L-inf error {norm} Time Taken: {end-start}")
          
          filename = f'masked_adv_{i}_{eps}.jpg'

          celeb_save_path = f'/content/data/celebatest_adv/{"-".join(ground_label.split(" "))}'

          if not os.path.exists(celeb_save_path):
            os.mkdir(celeb_save_path)
          
          if img_show:
            plt.imshow(adv_image)

            fig1 = plt.gcf()
            plt.show()
            fig1.savefig(f'{celeb_save_path}/{filename}')
          else:
            image.save_img(f'{celeb_save_path}/{filename}', adv_image)

          model = app.models.get('celeb-v1.3')
          response = model.predict_by_filename(f'{celeb_save_path}/{filename}')
          
          name = response['outputs'][0]['data']['regions'][0]['data']['concepts'][0]['name']
          if debug:
            print("API response:" + name)
          
          if name == ground_label:
            history[norm] += 1
            
        except Exception as e:
          print("Error:", e)
        attack.max_iter = iter_step

  return history

def generate_adv_masked_image():
  history = run_fgsm_attacks(
      classifier, 
      target_image, 
      eps_range=list(range(13)), 
      mask_width=30,
      masked=True,
      feature_extractor='mtcnn')

if __name__ == "__main__":
  print('loaded patchface')