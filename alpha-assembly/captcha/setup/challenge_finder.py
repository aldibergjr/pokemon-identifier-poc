# which pokemon is this?
import keras_ocr
import matplotlib.pyplot as plt
import os
import string
import cv2
import numpy as np


DETECTOR_PATH = 'C:/Users/bergc/pxg-bot/alpha-assembly/assets/recognizer/model_recognizer.h5'

    # Carrega o detector com os pesos treinados
detector = keras_ocr.detection.Detector()
detector.model.load_weights(DETECTOR_PATH)

# Carrega o recognizer com o alfabeto padrão do keras_ocr
recognizer = keras_ocr.recognition.Recognizer()
recognizer.compile()
print("Recognizer loaded")

# Monta o pipeline com detector + recognizer
pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)

def find_challenge(frame):
    # Get the absolute path to the main.py file
    # main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Construct the path to the model file relative to main.py
    

    # Lê e processa a imagem
    prediction_groups = pipeline.recognize([frame])  # lista com 1 elemento (batch size 1)

    # drawn = keras_ocr.tools.drawBoxes(
    #     image=frame, boxes=prediction_groups[0], boxes_format='predictions'
    # )

    # Mostra resultado
    # keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
    # plt.imshow(drawn)
    # # plt.axis('off')
    # plt.show()

    # Também imprime os textos reconhecidos
    for text, box in prediction_groups[0]:
        print('Predicted:', text)

    return text in prediction_groups[0]

