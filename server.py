from flask import Flask, request, send_file
import traceback
import os

app = Flask(__name__)

from patchface import generate_adv_masked_image, patch_input_image
from werkzeug.utils import secure_filename
import cv2 as cv
import logging

@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/patchface", methods=['POST'])
def patch_uploaded_file():
    logging.info(request.files)

    try:
        f = request.files.get("image")

        if f is None:
            return "File invalid", 400

        file_path = f"/tmp/{secure_filename(f.filename)}"
        f.save(file_path)
        adv_image = None

        if os.environ.get('GENERATE_PATCH') == True:
            logging.info('Generating image using fgsm')
            adv_image = generate_adv_masked_image(file_path)
        else:
            adv_image = patch_input_image(file_path)

        logging.info(adv_image)

        adv_path = f'/tmp/adv-{secure_filename(f.filename)}'
        
        cv.imwrite(adv_path, adv_image)
        return send_file(adv_path, 'img/jpeg', attachment_filename=secure_filename(f.filename))

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return "Internal Server Error", 500
