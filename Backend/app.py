from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2

# Inicializa la app de Flask
app = Flask(__name__)

# Ruta al modelo (asegúrate de tener el modelo correcto y la ruta configurada)
model_path = 'Backend/modelo.h5'  # Ruta al archivo .h5
model = load_model(model_path)

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo manual de las clases que el modelo puede predecir (ajusta esto según tu modelo)
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

@app.route('/')
def home():
    return 'API de Detección de Señales de LSM funcionando'

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del cuerpo de la solicitud
    data = request.get_json()
    keypoints = np.array(data['keypoints']).reshape(1, -1)  # Convertir las coordenadas en un array numpy
    prediction = model.predict(keypoints)  # Hacer la predicción

    # Obtener la clase más probable
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]

    # Devolver la predicción
    return jsonify({'prediction': predicted_class})

# Función para procesar la imagen y obtener las coordenadas de las manos
def process_image(image):
    # Convertir la imagen a RGB (MediaPipe usa RGB en lugar de BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints)  # Retornar las coordenadas de la mano

    return None  # Si no se detecta ninguna mano

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


