# import pyautogui
import numpy as np
from captcha.flow_trigger import FlowTrigger
import time
import cv2
import numpy as np
import os
from PIL import Image
from consts import all_pokemon_names
from captcha.solve.pokeball_detector import PokeballDetector

from captcha.setup.find_pokemon_position import PokemonFinder

capture_region = (0, 0, 1920, 1080)

def main():
    # image = cv2.imread('assets/tests/shuffle_in_proccess.png')
    # pokeball_detector = PokeballDetector("assets/templates/pokeball_template2.png")
    # pokeball_detector.find_pokeballs(image)
    # Open the video file
    video_path = "assets/tests/video/video_test.mp4"  # Replace with your video path
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    flow_trigger = FlowTrigger()

    # Process each frame from the video
    frame_count = 0
    total_elapsed_time = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video file")
            break
       
        # Process every 30th frame (adjust this number based on your needs)
        # also calculate total elapsed time
        if frame_count % 5 == 0:
            print(f"Processing frame {frame_count}")
            start_time = time.time()
            flow_trigger.verify_and_execute_captcha_state(frame)
            end_time = time.time()
            total_elapsed_time += end_time - start_time
            print(f"Total run time for frame {frame_count}: {end_time - start_time} seconds")
            print(f"Total elapsed time: {total_elapsed_time} seconds")
            
        frame_count += 1
        time.sleep(0.016)  # Small delay to prevent overwhelming processing
    
    # Release the video capture object
    cap.release()

    # CURR STEP
    # screen_image = cv2.imread('assets/tests/captcha6.png')

    # if screen_image is None:
    #     print("Erro ao carregar a imagem da tela.")
    #     return

if __name__ == "__main__":
    main()