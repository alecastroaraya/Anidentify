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

def detectar_cara(filename, cascade_file = "lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: no encontrado" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # opciones de detector de caras
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        cv2.imshow("AnimeFaceDetect", face)
        cv2.waitKey(0)
        cv2.imwrite("out.png", face)
        return True
    else:
        return False

def predecir_personajes(img_path):
    IMAGE_SIZE = (96,96)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = normalization_layer(img_array)

    class_names = ('Aegis-Persona', 'Ai Hayasaka-Kaguya Sama Love Is War', 'Aisaka Taiga-Toradora', 'Albedo-Overlord', 'Amelia Watson-Hololive', 'Aqua-Konosuba', 'Arcueid Brunestud-Tsukihime', 'Astolfo-Fate', 'Asuka Langley Souryuu-Neon Genesis Evangelion', 'Asuna Yuuki-Sword Art Online', 'C.C-Code Geass', 'Chika Fujiwara-Kaguya Sama Love Is War', 'Cirno-Touhou', 'Emilia-Re Zero', 'Flandre Scarlet-Touhou', 'Fubuki-One Punch Man', 'Gawr Gura-Hololive', 'Giorno Giovanna-JoJos Bizarre Adventure', 'Haruhi Suzumiya-The Melancholy of Haruhi Suzumiya', 'Hatsune Miku-Vocaloid', 'Hifumi Takimoto-New Game!', 'Hinata Hyuuga-Naruto', 'Illyasviel Von Einzbern-Fate', 'Itsuki Nakano-Quintessential Quintuplets', 'Jonathan Joestar-Jojos Bizarre Adventure', 'Kagami Hiiragi-Lucky Star', 'Kaguya Shinomiya-Kaguya Sama Love Is War', 'Kallen Stadtfeld-Code Geass', 'Keqing-Genshin Impact', 'Kirito-Sword Art Online', 'Konata Izumi-Lucky Star', 'Kotobuki Tsumugi-K ON!', 'Kurisu Makise-Steins Gate', 'Lelouch Lamperouge-Code Geass', 'Mai Sakurajima-Rascal Does Not Dream of Bunny Girl Senpai', 'Maka Albarn-Soul Eater', 'Maki Nishikino-Love Live', 'Megumin-Konosuba', 'Miku Nakano-Quintessential Quintuplets', 'Misato Katsuragi-Neon Genesis Evangelion', 'Nami-One Piece', 'Nezuko Kamado-Demon Slayer', 'Nia Teppelin-Gurrenn Lagann', 'Nico Robin-One Piece', 'Nino Nakano-Quintessential Quintuplets', 'Nozomi Tojo-Love Live', 'Ouro Kronii-Hololive', 'Paimon-Genshin Impact', 'Ram-Re Zero', 'Raphtalia-Rising of the Shield Hero', 'Rei Ayanami-Neon Genesis Evangelion', 'Reimu Hakurei-Touhou', 'Rem-Re Zero', 'Rikka Takanashi-Love, Chunibyo and Other Delusions!', 'Rin Tohsakaka-Fate', 'Ritsu Tainaka-K ON!', 'Saber-Fate', 'Sakura Matou-Fate', 'Sakuya Izayoi-Touhou', 'Scarlet Remilia-Touhou', 'Shiki Ryougi-Garden of Sinners', 'Shinji Ikari-Neon Genesis Evangelion', 'Shinobu Kochou-Demon Slayer', 'Shirley Fenette-Code Geass', 'Shouko Komi-Komi Cant Communicate', 'Sinon-Sword Art Online', 'Suzaku Kururugi-Code Geass', 'Yoko Littner-Gurrenn Lagann', 'Yui Hirasawa-K ON!', 'Zero Two-Darling In the Franxx')

    plt.imshow(img_array[0,:,:,:])
    plt.axis('off')
    plt.show()

    predicciones = modelo_cargado.predict(img_array)
    probabilidades = tf.nn.softmax(predicciones[0])
    top_5_indices = tf.math.top_k(probabilidades, k=5).indices.numpy()
    top_5_probabilidades = tf.gather(probabilidades, top_5_indices).numpy() * 100

    nombres_personajes_anime = [class_names[i] for i in top_5_indices]

    resultados = []
    for nombre_completo, probabilidad in zip(nombres_personajes_anime, top_5_probabilidades):
        nombre_personaje, nombre_anime = nombre_completo.split("-")
        nombre_personaje = nombre_personaje.strip()
        nombre_anime = nombre_anime.strip()
        resultados.append((nombre_personaje, nombre_anime, probabilidad))

    return resultados



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

        cara_encontrada = detectar_cara(filename)
        if cara_encontrada:
            resultados = predecir_personajes("out.png")
            return render_template('upload.html', resultados=resultados, filename=file.filename)
        else:
            flash('Error: No se detectaron caras en la imagen.', 'error')
            return render_template('upload.html')
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