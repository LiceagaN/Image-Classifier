# Image Classifier Project

Este proyecto utiliza PyTorch para construir, entrenar y evaluar un modelo de clasificación de imágenes basado en redes neuronales profundas. Está diseñado para clasificar flores en 102 categorías, aprovechando arquitecturas de red preentrenadas como DenseNet y AlexNet.

## Contenido del Proyecto

### Archivos

1. **Image Classifier Project.ipynb**  
   Un notebook interactivo que incluye pasos para:
   - Cargar y preprocesar los datos.
   - Construir y entrenar el modelo de clasificación.
   - Evaluar el modelo en datos de prueba.
   - Guardar el modelo entrenado como un checkpoint.

2. **train.py**  
   Un script Python que permite entrenar el modelo desde la línea de comandos. Ofrece opciones configurables para los hiperparámetros, la arquitectura del modelo y el uso de GPU. Este script también guarda el modelo entrenado como un checkpoint.

3. **predict.py**  
   Un script Python que permite realizar predicciones sobre imágenes individuales utilizando un modelo previamente entrenado. Entre sus funcionalidades, incluye:
   - Cargar un checkpoint para restaurar el modelo entrenado.
   - Procesar imágenes para que sean compatibles con el modelo.
   - Predecir las categorías de las imágenes, retornando las probabilidades y los nombres de las categorías más probables.

## Requisitos Previos

Asegúrate de tener instalados los siguientes paquetes y bibliotecas antes de comenzar:

- Python >= 3.7
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

Puedes instalar los requisitos ejecutando:

```bash
pip install torch torchvision numpy matplotlib pillow
```
## Como Usar
Abre el archivo **Image Classifier Project.ipynb** en un entorno como Jupyter Notebook.
Sigue los pasos del notebook para:

    Cargar el conjunto de datos.
    Entrenar el modelo.
    Evaluar el modelo y realizar predicciones.
    Guardar el modelo entrenado.
El notebook proporciona un flujo guiado para entrenar y evaluar el modelo, sin necesidad de escribir código adicional.

### Usando el Script de Entrenamiento
El archivo** train.py** ofrece varias opciones configurables desde la línea de comandos. Puedes ejecutar el script con los siguientes parámetros:
```bash
python train.py flowers --save_dir checkpoints --arch densenet169 --learning_rate 0.003 --epochs 20 --hidden_unit_per_layer 1024 512 --gpu
```
flowers: Directorio de datos que contiene las carpetas train, valid y test.
--save_dir: Directorio donde se guardará el checkpoint del modelo entrenado.
--arch: Arquitectura del modelo (densenet169 o alexnet).
--learning_rate: Tasa de aprendizaje.
--epochs: Número de épocas de entrenamiento.
--hidden_unit_per_layer: Número de unidades en las capas ocultas.
--gpu: Usar GPU para entrenar (habilitado por defecto).

### Usando el Script de Predicción

El archivo **predict.py** permite predecir categorías de flores en imágenes individuales. Puedes ejecutar el script con los siguientes parámetros:
```bash
python predict.py path/to/image checkpoint.pth --top_k 5 --categories cat_to_name.json --gpu
```

path/to/image: Ruta de la imagen para la predicción.
checkpoint.pth: Ruta al checkpoint del modelo entrenado.
--top_k: Número de categorías principales para mostrar (por defecto: 5).
--categories: Archivo JSON que mapea las etiquetas de categorías con sus nombres.
--gpu: Habilita GPU para predicciones (habilitado por defecto).

### Datos de Entrenamiento

Los datos deben estar organizados en carpetas como sigue:
```bash
flowers/
    train/
    valid/
    test/
```
