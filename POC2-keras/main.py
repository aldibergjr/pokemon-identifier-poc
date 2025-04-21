import keras_ocr
import matplotlib.pyplot as plt
import os
import string

# Caminho do modelo treinado
DETECTOR_PATH = 'assets/checkpoints/detector_2025-04-21--10-02-54.901062_final_manual_save.h5'  # ajuste se necessário
TEST_IMAGE_PATH = 'assets/images/app_tests/where_is_arcanine_city_bg.png'

# Carrega o detector com os pesos treinados
detector = keras_ocr.detection.Detector()
detector.model.load_weights(DETECTOR_PATH)

# Carrega o recognizer com o mesmo alfabeto usado no treino
recognizer = keras_ocr.recognition.Recognizer(alphabet=''.join(sorted(set(string.digits + string.ascii_letters + '!?. '))))
recognizer.compile()

# Monta o pipeline com detector + recognizer
pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)

# Lê e processa a imagem
image = keras_ocr.tools.read(TEST_IMAGE_PATH)
prediction_groups = pipeline.recognize([image])  # lista com 1 elemento (batch size 1)

# Mostra resultado
keras_ocr.tools.drawAnnotations(image=image, predictions=prediction_groups[0])
plt.imshow(image)
plt.axis('off')
plt.show()

# Também imprime os textos reconhecidos
for text, box in prediction_groups[0]:
    print(f"Reconhecido: '{text}' em {box}")
