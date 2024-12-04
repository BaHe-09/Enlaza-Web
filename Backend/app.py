import os
import time
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2

# Ruta para el modelo
model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5') # Ruta local cuando el script no está empaquetado

# Inicializar Flask
app = Flask(__name__)

# Intentar cargar el modelo de Keras y manejar el error si el formato es incorrecto
try:
    model = load_model(model_path, compile=False)
    print("Modelo cargado correctamente.")
except ValueError as e:
    print(f"Error al cargar el modelo: {e}")
    sys.exit(1)  # Terminar el script si el modelo no se puede cargar

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo manual de las clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')  # Renderizar la página principal

# Ruta de predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decodificar la imagen en base64
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convertir la imagen a RGB para MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    class_label = ""  # Predicción de la seña
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints, verbose=0)
            class_index = np.argmax(prediction)
            class_label = class_names[class_index]

    return jsonify({"predicted_class": class_label})

# Iniciar el servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


