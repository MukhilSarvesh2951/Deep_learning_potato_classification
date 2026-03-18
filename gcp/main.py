import functions_framework
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import flask

BUCKET_NAME = 'mukhil-tf-models'
class_names = ['Early Blight', 'Late Blight', 'Healthy']

model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

@functions_framework.http
def predict(request):
    global model

    if model is None:
        try:
            download_blob(BUCKET_NAME, 'models/potatoes.weights.h5', '/tmp/potatoes.weights.h5')
            model = build_model()
            model.load_weights('/tmp/potatoes.weights.h5')
        except Exception as e:
            model = None  # ✅ reset so next request tries again
            return flask.jsonify({"error": str(e)}), 500

    image = request.files["file"]
    img = Image.open(image).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(float(np.max(predictions[0])), 2)

    return flask.jsonify({
        "predictions": predicted_class,
        "confidence": confidence
    })