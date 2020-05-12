from flask import Flask, request, send_file
import traceback

app = Flask(__name__)

from patchface import generate_adv_masked_image
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

        adv_image = generate_adv_masked_image(file_path)

        logging.debug(adv_image)

        cv.imwrite(file_path, adv_image)
        
        cv.imwrite(file_path, cv.cvtColor(cv.imread(file_path), cv.COLOR_BGR2RGB))
        return send_file(file_path, 'img/jpeg', attachment_filename=secure_filename(f.filename))

    except Exception as e:
        tb = traceback.format_exc()
        logging.error(tb)
        return "Internal Server Error", 500
