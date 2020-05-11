from flask import Flask, request
app = Flask(__name__)

from patchface import generate_adv_masked_image
from werkzeug.utils import secure_filename

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/patchface')
def patch_uploaded_file():
  if request.method == 'POST':
    f = request.files['image']
    file_path = f'/tmp/{secure_filename(f.filename)}'
    f.save(file_path)

    generate_adv_masked_image()
