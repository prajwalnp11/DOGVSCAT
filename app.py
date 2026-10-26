
import os
import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "dog_cat_classifier.h5"
)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150, 150))

    image_array = np.asarray(image, dtype=np.float32)

    # Assumes training images were normalized to 0–1.
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_data = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select an image."
        else:
            try:
                with Image.open(uploaded_file.stream) as image:
                    image = image.convert("RGB")

                    # Keep the uploaded image for display.
                    preview = image.copy()
                    preview.thumbnail((700, 500))

                    buffer = BytesIO()
                    preview.save(buffer, format="JPEG")
                    image_data = base64.b64encode(
                        buffer.getvalue()
                    ).decode("utf-8")

                    # Prepare image for the model.
                    image_array = preprocess_image(image)

                score = float(
                    model.predict(image_array, verbose=0)[0][0]
                )

                # ASSUMPTION: sigmoid output >= 0.5 means Dog.
                if score >= 0.5:
                    label = "Dog"
                    emoji = "🐶"
                else:
                    label = "Cat"
                    emoji = "🐱"

                prediction = {
                    "label": label,
                    "emoji": emoji,
                    "score": f"{score:.4f}",
                    "confidence": f"{max(score, 1 - score) * 100:.1f}%"
                }

            except (UnidentifiedImageError, OSError, ValueError):
                error = "Please upload a valid image file."

    return render_template(
        "index.html",
        prediction=prediction,
        image_data=image_data,
        error=error
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
