# Anidentify

## Link del Demo:
**Link de la aplicación:** [Anidentify](placeholder)

## Tabla de contenidos

- [Acerca del app](#acerca-del-app)
- [Tecnologías utilizadas](#tecnologias-utilizadas)
- [Créditos](#creditos)
- [Licencia](#licencia)

## Acerca del app
Anidentify es una aplicación web que permite al usuario subir una imagen con un personaje de animé. Luego de esto, identifica el personaje y muestra el nombre completo y de qué franquicia proviene, junto con la probabilidad de que sea ese personaje y otras opciones de personaje menos probables.


## Tecnologias utilizadas
Se utilizaron las siguientes tecnologías para desarrollar la aplicación:
- **Flask** (Servidor del backend)
- **JavaScript/HTML/Bootstrap** (Frontend)
- **Oracle Cloud** (Alojamiento de la aplicación web)
- **Python** (Lenguaje de programación)

## Uso del app
Para consumidores, simplemente accede al link del sitio web y sube una imagen. Luego el sitio web te dirá la información del personaje.

Para desarrolladores, corre `python app.py` para correr localmente el sitio web y accederlo en `127.0.0.1:5000`. Además, puede utilizar el cuaderno `cuaderno_entrenamiento_modelo.ipynb` para entrenar, evaluar y correr el modelo utilizado en este proyecto. El folder train tiene las imagenes de entrenamiento, test tiene las imagenes para evaluar el modelo, y validation para la validacion del modelo.

## Creditos
**Paper fundamento:**
[AniWho : A Quick and Accurate Way to Classify Anime Character Faces in Images](https://paperswithcode.com/paper/aniwho-a-quick-and-accurate-way-to-classify) de Martinus Grady Naftali, Jason Sebastian Sulistyawan, Kelvin Julian.

**Datasets utilizados:**
[Tagged Anime Illustrations](https://www.kaggle.com/datasets/mylesoneill/tagged-anime-illustrations) de Myles O'Neill.

[Anime Face Dataset by Character Name](https://paperswithcode.com/dataset/anime-face-dataset-by-character-name) de Naftali et al.

**Recursos utilizados para la detección de caras:**
[LBPCascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) se utilizó como archivo de cascade de OpenCV para poder detectar específicamente caras de tipo animé en las imágenes.

**Recursos utilizados para el HTML/CSS:**
[Bootstrap](https://getbootstrap.com/) se utilizó para el frontend de la página web.

## Licencia

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)