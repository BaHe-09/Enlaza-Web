from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image
import io
import base64

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta al modelo (ajustar para que funcione en Render)
model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5')  # Modificar según la ruta correcta en Render
model = load_model(model_path)

# Mapeo de las clases que el modelo puede predecir
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Ruta principal
@app.route('/')
def home():
    return "API de Detección de LSM funcionando"

# Ruta para recibir y procesar la imagen
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Obtener los datos enviados desde el frontend
        image_data = data['image']  # La imagen en base64

        # Decodificar la imagen de base64
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        img = np.array(img)

        # Procesar la imagen
        processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        processed_img = cv2.resize(processed_img, (224, 224))  # Redimensionar para el modelo
        processed_img = np.expand_dims(processed_img, axis=0)  # Agregar dimensión para el batch

        # Hacer la predicción
        prediction = model.predict(processed_img)
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]

        return jsonify({'prediction': predicted_class})  # Devolver la predicción

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Configurado para localhost

