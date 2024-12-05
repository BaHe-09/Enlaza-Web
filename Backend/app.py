import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import json

app = Flask(__name__)

# Cargar el modelo Keras previamente entrenado
model= os.path.join(os.getcwd(), 'Backend', 'modelo.h5')

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route('/')
def home():
    return "Servidor Funcionando"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    points = data.get('points')  # Lista de puntos clave (landmarks)
    
    if not points:
        return jsonify({'error': 'No se recibieron puntos.'}), 400

    # Convertir los puntos a un array numpy (deben estar en formato adecuado para el modelo)
    input_data = np.array(points).flatten()
    
    # Normalizar o hacer cualquier transformación necesaria para tu modelo
    input_data = np.expand_dims(input_data, axis=0)

    # Realizar la predicción
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)  # Ajusta según tu modelo

    # Devolver la predicción como texto (ajusta esto dependiendo de tu modelo)
    prediction_label = f'Clase {predicted_class[0]}'

    return jsonify({'prediction': prediction_label})

@app.route('/video', methods=['GET'])
def video():
    try:
        return jsonify({"message": "Video endpoint working."})
    except Exception as e:
        return jsonify({"error": "No se pudo acceder a la cámara."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


