from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import os

app = Flask(__name__)

# Ruta al modelo (en la carpeta Backend). Usamos os.path.join para una ruta más confiable
model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5')  # Ruta del modelo en la carpeta Backend
model = load_model(model_path)

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

@app.route('/')
def home():
    return render_template('index.html')  # Servir el archivo index.html

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Obtener las coordenadas de las manos
    image_data = data['image']
    image = decode_image(image_data)
    keypoints = process_image(image)

    if keypoints is not None:
        keypoints = np.array(keypoints).reshape(1, -1)
        prediction = model.predict(keypoints)
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]
        return jsonify({'prediction': predicted_class})

    return jsonify({'prediction': 'No se detectó ninguna mano'})

def decode_image(image_data):
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return keypoints
    return None

# Asegurarse de usar el puerto de Render
if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)  # Obtener el puerto de Render (por defecto 5000)
    app.run(debug=True, host='0.0.0.0', port=port)  # Ejecutar Flask con el puerto proporcionado
