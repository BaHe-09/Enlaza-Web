import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

# Ruta al modelo
model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5')

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo de Keras
model = load_model(model_path, compile=False)

# Mapeo manual de las clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

@app.route('/')
def home():
    return render_template('index.html')  # Renderizar la página principal

# Ruta para recibir la imagen y hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Recibir la imagen en base64
    
    # Decodificar la imagen base64
    img_data = base64.b64decode(data['image'])
    img = Image.open(BytesIO(img_data))

    # Preprocesar la imagen antes de pasarla al modelo
    img = img.resize((224, 224))  # Ajusta el tamaño según el tamaño de entrada de tu modelo
    img_array = np.array(img) / 255.0  # Normalización si es necesario

    # Convertir la imagen a un formato compatible con el modelo (reshape)
    img_array = img_array.reshape(1, -1)

    # Hacer la predicción
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]

    # Devolver la predicción al frontend
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
