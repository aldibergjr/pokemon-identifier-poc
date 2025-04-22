# which pokemon is this?
import keras_ocr
import matplotlib.pyplot as plt
import os
import string
import cv2
import numpy as np
from consts import all_pokemon_names


DETECTOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets\\recognizer\\model_recognizer.h5'
)

# Carrega o detector com os pesos treinados
detector = keras_ocr.detection.Detector()
detector.model.load_weights(DETECTOR_PATH)

# Carrega o recognizer com o alfabeto padrão do keras_ocr
recognizer = keras_ocr.recognition.Recognizer()
recognizer.compile()
print("Recognizer loaded")

# Monta o pipeline com detector + recognizer
pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)

def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    This is a measure of the minimum number of single-character edits required to change one string into the other.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_closest_pokemon(ocr_text, max_distance=3):
    """
    Find the closest Pokemon name from the constants file that matches the OCR text.
    
    Args:
        ocr_text (str): The text detected by OCR
        max_distance (int): Maximum Levenshtein distance to consider a match
        
    Returns:
        str: The closest matching Pokemon name, or None if no match is found
    """
    # Clean the OCR text - remove extra spaces, convert to lowercase
    ocr_text = ocr_text.strip().lower()
    
    # If the OCR text is empty, return None
    if not ocr_text:
        return None
    
    # Calculate the Levenshtein distance between the OCR text and each Pokemon name
    closest_match = None
    min_distance = float('inf')
    
    for pokemon_name in all_pokemon_names:
        # Convert Pokemon name to lowercase for comparison
        pokemon_name_lower = pokemon_name.lower()
        
        # Calculate the Levenshtein distance
        distance = levenshtein_distance(ocr_text, pokemon_name_lower)
        
        # Update the closest match if this is closer
        if distance < min_distance:
            min_distance = distance
            closest_match = pokemon_name
    
    # Return the closest match if it's within the maximum distance
    if min_distance <= max_distance:
        return closest_match
    
    return None

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

    # Get the third text (Pokemon name) from OCR
    ocr_text = prediction_groups[0][2][0]
    print('OCR detected Pokemon:', ocr_text)
    
    # Find the closest matching Pokemon name
    pokemon_name = find_closest_pokemon(ocr_text)
    
    if pokemon_name:
        print('Matched to Pokemon:', pokemon_name)
        return pokemon_name
    else:
        print('Could not match to a valid Pokemon name')
        return ocr_text  # Return the original OCR text if no match is found

