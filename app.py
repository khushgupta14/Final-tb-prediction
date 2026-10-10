import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tb_model.h5")

app = Flask(__name__)
model = load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="No file selected"), 400

    # Read and preprocess image in-memory
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = float(model.predict(img_array)[0][0])
    tb_percent = round(prob * 100, 2)
    normal_percent = round((1.0 - prob) * 100, 2)

    return jsonify(normal=normal_percent, tb=tb_percent)

if __name__ == "__main__":
    app.run(debug=True)
