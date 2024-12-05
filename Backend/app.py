import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Ruta al modelo
model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5')

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo de Keras
model = load_model(model_path, compile=False)

# Mapeo manual de las clases (de acuerdo a tu entrenamiento)
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

@app.route('/')
def home():
    return "Servidor funcionando"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Obtener los puntos clave de la mano
    points = np.array(data['points'])

    # Asegurarse de que los puntos tengan la forma adecuada para el modelo
    # Si el modelo espera un vector plano, puedes aplanar los puntos
    points = points.flatten()

    # Normalizar los puntos (si es necesario según el entrenamiento del modelo)
    points = points / np.max(points)

    # Hacer la predicción con el modelo
    prediction = model.predict(np.expand_dims(points, axis=0))
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]

    # Devolver la predicción al frontend
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

