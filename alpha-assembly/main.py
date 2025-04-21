# import pyautogui
import numpy as np
from captcha.flow_trigger import verify_and_execute_captcha_state
import time
import cv2
import numpy as np

from captcha.setup.find_pokemon_position import PokemonFinder

capture_region = (0, 0, 1920, 1080)

def main():
    # change this with a fori for the images captcha1.png, captcha2.png, captcha3.png
    # for i in range(1, 8):

    #     # Capture screen
    #     # screenshot = pyautogui.screenshot(region=capture_region)
    #     # frame = np.array(screenshot)
    #     frame = cv2.imread(f'assets/tests/captcha{i}.png')
    #     # measure total run time
    #     start_time = time.time()
    #     verify_and_execute_captcha_state(frame)
    #     end_time = time.time()
    #     print(f"Total run time: {end_time - start_time} seconds")
    #     time.sleep(3)
    screen_image = cv2.imread('assets/tests/captcha2.png')

    if screen_image is None:
        print("Erro ao carregar a imagem da tela.")
        return

    # Instancia o detector
    finder = PokemonFinder('Nidoqueen')  # troque pelo nome do Pokémon que você quer testar

    # Encontra as posições
    matches = finder.find_on_screen(screen_image)

    # Desenha as detecções
    for (x, y, r, score) in matches:
        cv2.circle(screen_image, (x, y), r, (0, 255, 0), 2)
        cv2.putText(screen_image, f"{score:.2f}", (x - r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostra a imagem com as marcações
    cv2.imshow("Detecções de Pokémon", screen_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()