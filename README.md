# Proyecto de Análisis de Imágenes para Detección de Hormigas en Plantas
**Este proyecto tiene como objetivo desarrollar un sistema de análisis de imágenes en Python para la detección de bordes de hormigas negras en plantas. Utiliza varios algoritmos de procesamiento de imágenes para identificar y delinear los bordes de las hormigas en las imágenes de las plantas.**

![Screenshot of a comment on a GitHub issue showing an image, added in the Markdown, of an Octocat smiling and raising a tentacle.](/Imagenes/Captura.JPG)

## Algoritmos Utilizados
- Subir la Imagen: Carga la imagen de la planta con las hormigas.
- Canales RGB: Analiza los diferentes canales RGB de la imagen para resaltar las características.
- Filtrado Gaussiano: Aplicado para suavizar las imágenes y reducir el ruido.
- Transformada de Sobel: Utilizada para detectar los bordes en las imágenes.
- Binarización: Aplicada para convertir la imagen en una imagen binaria y resaltar los bordes.
- Filtro Laplaciano Negativo: Aplicado para realzar los bordes.
- Operación OR: Combina los bordes resaltados para obtener un resultado final.
## Requisitos
- Python 3.x
- Bibliotecas de Python:
- OpenCV
- NumPy
## Instalación
- Clona este repositorio en tu máquina local usando git clone.
- Asegúrate de tener Python instalado en tu sistema.
- Instala las dependencias utilizando pip:
- Copy code pip install opencv-python numpy
- Ejecuta el script main.py para abrir la interfaz y comenzar el análisis de imágenes.

## Uso
- Coloca las imágenes de las plantas con hormigas en la carpeta imagenes/.
- Ejecuta el script main.py.
- Selecciona la imagen a procesar.
- Analiza los canales RGB para identificar las características.
- Aplica el filtrado Minimo para suavizar la imagen.
- Realiza la binarización para resaltar los bordes.
- Aplica el Filtro Laplaciano Negativo para realzar los bordes.
- Combina los bordes resaltados con la operación OR para obtener un resultado final.
