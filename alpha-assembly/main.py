# import pyautogui
import numpy as np
from captcha.flow_trigger import verify_and_execute_captcha_state
import time
import cv2
import numpy as np

capture_region = (0, 0, 1920, 1080)

def main():
    # change this with a fori for the images captcha1.png, captcha2.png, captcha3.png
    for i in range(1, 8):

        # Capture screen
        # screenshot = pyautogui.screenshot(region=capture_region)
        # frame = np.array(screenshot)
        frame = cv2.imread(f'assets/tests/captcha{i}.png')
        # measure total run time
        start_time = time.time()
        verify_and_execute_captcha_state(frame)
        end_time = time.time()
        print(f"Total run time: {end_time - start_time} seconds")
        time.sleep(3)

if __name__ == "__main__":
    main()