import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import base64
import io

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo de Keras
model_path = os.path.join(os.getcwd(), 'modelo.h5')
try:
    model = load_model(model_path, compile=False)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo manual de las clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Ruta principal de la aplicación web
@app.route('/')
def home():
    return render_template('index.html')  # Renderizar la página principal

# Función para capturar el video y procesarlo
def generate_frames():
    cap = cv2.VideoCapture(0)  # Captura de la cámara local

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return jsonify({"error": "No se pudo acceder a la cámara."}), 500

    while True:
        success, image = cap.read()
        
        if not success or image is None:
            continue

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

        # Mostrar los puntos de la mano en la imagen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Dibujar puntos de las manos

        # Convertir la imagen a formato JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Enviar el frame como un flujo de bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Ruta para mostrar el video en el navegador (streaming)
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para recibir la imagen y hacer la predicción
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



