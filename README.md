# Patchy-McPatchface

Apply adversarial masks to profile photos to protect your online and offline privacy.

Media uploaded on social networking sites or on the web is frequently mined for facial recognition training and mining. 
In order to retain some semblance of privacy, this project seeks to apply adversarial noise on photos of faces, to fool well known facial recognition algorithms.

There are two ways to combat facial recognition software - Attacks during training time and prediction time. 
Since facial recognition software are largely black box models, it is hard to attack these models. 
This project assumes a "best effort" attack, by attacking prediction time white box models instead.

## Metrics
- Fools [Clarifai](https://clarifai.com/) Celebrity Recognition with an L-infinity norm of 11. At this level, the adversarial patch
is indistinguishable.
- Fools Google Reverse Image Search with an L-infinity norm of 13.

## Disclaimer
This software has *not* had extensive testing. Please use at your own risk. It's provided as a best effort software to combat facial recognition.

## Run Instructions

- Use pipenv: pipenv install
- Run script on a profile photo: `python3 patchface.py --image-path <path-to-your-file>`.

The script loads Deepface, a facial verification model. Using GPU can substantially speed up the patch generation.

## Roadmap
 - Create Flask server for generating patches
 - Offload adversarial generation to browser side using tensorflow.js

## Contributions
Contributions are very welcome! Please open an issue for contributions, bugs and questions.


