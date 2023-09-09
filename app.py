from flask import Flask, render_template, request, send_from_directory, flash
import os
import tensorflow as tf
import cv2
import sys
from glob import glob
import itertools
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Especifica la ruta donde esta el modelo
ruta_guardado = "modelo_guardado"

# Carga el modelo
modelo_cargado = tf.keras.models.load_model(ruta_guardado)

app = Flask(__name__)

# Configuración para subir archivos
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['SECRET_KEY'] = 'aaaaaaasdasdasdasdsadaaaaaaaaasdasdasdasdasa'

# Función para verificar extensiones permitidas
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('upload.html')

def build_dataset(subset):
  # Esta funcion recibe un directorio con clases que contienen imagenes y lo preprocesa para usar como dataset
  data_dir = "train"
  IMAGE_SIZE = (96,96)

  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.18,
      subset=subset,
      label_mode="categorical",

      seed=321,
      image_size=IMAGE_SIZE,
      batch_size=1)

def predecir_personajes(img_path):
    # Se carga y prepara la imagen para la predicción

    IMAGE_SIZE = (96,96)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Se crea un batch
    img_array = normalization_layer(img_array)

    train_ds = build_dataset("training")
    class_names = tuple(train_ds.class_names)

    plt.imshow(img_array[0,:,:,:])
    plt.axis('off')
    plt.show()

    # Realiza la predicción
    predicciones = modelo_cargado.predict(img_array)

    # Obtiene el índice de la predicción más probable
    index_predecido = np.argmax(predicciones[0])
    confianza = 100 * tf.nn.softmax(predicciones[0])[index_predecido]
    valor_numerico_confianza = confianza.numpy()

    nombre_personaje_anime = class_names[index_predecido]
    # Imprime el resultado
    print(f"Este personaje posiblemente es: {class_names[index_predecido]}")
    print(f"Con {confianza:.2f}% confianza.")

    # Separar el nombre del personaje y el nombre del anime
    nombre_personaje, nombre_anime = nombre_personaje_anime.split("-")

    # Eliminar espacios en blanco adicionales alrededor del nombre del personaje
    nombre_personaje = nombre_personaje.strip()

    # Eliminar espacios en blanco adicionales al inicio del nombre del anime
    nombre_anime = nombre_anime.strip()

    return nombre_personaje,nombre_anime,valor_numerico_confianza


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Especifica la ruta donde esta el modelo
        ruta_guardado = "modelo_guardado"

        # Carga el modelo
        modelo_cargado = tf.keras.models.load_model(ruta_guardado)
        nombre_personaje,nombre_anime,confianza = predecir_personajes(filename)

        return render_template('upload.html', nombre_personaje=nombre_personaje,nombre_anime=nombre_anime, confianza=confianza, filename=file.filename)
    else:
        flash('Error: Por favor selecciona y sube un archivo.', 'error')
        return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/como-usar', methods=['GET','POST'])
def como_usar():
    return render_template('como-usar.html')

if __name__ == '__main__':
    app.run(debug=True)