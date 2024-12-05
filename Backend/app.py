import os
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Cargar el modelo al iniciar la aplicación
model = load_model('Backend/modelo.h5')  # Ajusta la ruta si es necesario

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Definir las clases del modelo (ajusta esto según las clases que hayas utilizado al entrenar el modelo)
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]
 # Cambia esto según tus clases

# Ruta principal que sirve la página de inicio
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar el video y predecir los signos
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Función que genera los frames del video
def generate_frames():
    import cv2

    # Configurar la cámara (Ajusta el índice si es necesario)
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convertir el frame a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el frame con MediaPipe
        results = hands.process(rgb_frame)

        # Si hay puntos clave detectados
        if results.multi_hand_landmarks:
            # Extraer los puntos clave de la mano
            landmarks = results.multi_hand_landmarks[0]
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

            # Preprocesar los puntos clave para la predicción (ajusta según el formato que requiera tu modelo)
            input_data = preprocess_points(points)  # Función de preprocesamiento
            input_data = np.expand_dims(input_data, axis=0)  # Ajuste de dimensiones para el modelo

            # Hacer la predicción
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_names[predicted_class]

            # Enviar la predicción al frontend
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')

        # Finalizar captura de video
    cap.release()

# Función para preprocesar los puntos de la mano antes de la predicción (ajusta según el modelo)
def preprocess_points(points):
    # Aquí puedes aplicar cualquier tipo de preprocesamiento a los puntos si es necesario.
    # Este es solo un ejemplo que normaliza los puntos.
    points = np.array(points)
    points = points.flatten()  # Aplana el arreglo de puntos
    return points

# Ruta para la predicción usando los puntos clave enviados desde el frontend
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'points' in data:
        points = np.array(data['points'])
        input_data = preprocess_points(points)  # Preprocesar puntos
        input_data = np.expand_dims(input_data, axis=0)  # Ajuste de dimensiones para el modelo

        # Realizar la predicción
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]

        # Enviar la predicción como respuesta
        return jsonify({'prediction': predicted_label})
    
    return jsonify({'error': 'No se recibieron puntos clave'}), 400

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

