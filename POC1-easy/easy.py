import easyocr
import re
import time
from PIL import Image
import mss
import numpy as np
import cv2
import os
from pathlib import Path
from datetime import datetime

# Initialize OCR reader
reader = easyocr.Reader(['pt'])

# Screen crop region (based on your screenshot)
# Adjust if needed: (left, top, right, bottom)
crop_box = (500, 100, 1500, 600)  # Adjusted to focus on the game area where Pokemon names appear

# Ensure debug folder exists
os.makedirs("debug", exist_ok=True)

def calibrate_crop_box():
    """
    Helper function to calibrate the crop box visually.
    Shows a window with the current crop region highlighted.
    Press 'q' to quit the calibration.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary screen
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        
        # Draw rectangle for crop region
        cv2.rectangle(img, 
            (crop_box[0], crop_box[1]), 
            (crop_box[2], crop_box[3]), 
            (0, 255, 0), 2)
        
        # Add text with coordinates
        coord_text = f"Crop box: {crop_box}"
        cv2.putText(img, coord_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the image
        cv2.imshow('Crop Box Calibration', img)
        while True:
            # Capture full screen
           
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image using HSV filtering to isolate yellow text
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise AND to keep only yellow text
    result = cv2.bitwise_and(image, image, mask=mask)

    # Convert result to grayscale and threshold it
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Increase size to help with recognition
    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Optional: Add a small amount of dilation to make text more prominent
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    return dilated

def template_match(image: np.ndarray, template: np.ndarray, threshold=0.8) -> bool:
    """
    Perform template matching and return True if template is found
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > threshold

def save_debug_image(image: np.ndarray, prefix: str):
    """
    Save debug images with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"debug/{prefix}_{timestamp}.png"
    cv2.imwrite(filename, image)
    print(f"Saved debug image: {filename}")

def extract_pokemon_name(image: Image.Image) -> str | None:
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Save original image
    # save_debug_image(img_array, "original")
    
    # Apply preprocessing
    processed_img = preprocess_image(img_array)
    
    # Save processed image
    # save_debug_image(processed_img, "processed")
    
    # Run OCR with settings optimized for the processed image
    results = reader.readtext(
        processed_img,
        paragraph=False,     # Treat each text area separately
        height_ths=0.5,     # More lenient height threshold
        width_ths=0.5,      # More lenient width threshold
        contrast_ths=0.1,   # Very lenient contrast threshold
        text_threshold=0.7  # Slightly higher confidence threshold for cleaner text
    )
    
    for _, text, conf in results:
        print(f"Detected text: {text} (confidence: {conf:.2f})")
        match = re.search(r"Onde está\s+(\w+)\??", text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def capture_screen_crop():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary screen
        screenshot = sct.grab(monitor)
        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        return img.crop(crop_box)

def main():
    # Ask if user wants to calibrate
    # choice = input("Do you want to calibrate the crop box? (y/n): ").lower()
    # if choice == 'y':
    #     print("Showing crop box calibration window...")
    #     print("Press 'q' to exit calibration mode")
    #     calibrate_crop_box()
        
    print("🕵️ Aguardando detecção de Pokémon... Pressione Ctrl+C para parar.")
    last_detected = None
    while True:
        cropped_img = capture_screen_crop()
        pokemon = extract_pokemon_name(cropped_img)

        if pokemon and pokemon != last_detected:
            print(f"✅ Pokémon detectado: {pokemon}")
            last_detected = pokemon

        time.sleep(2)  # Check every 2 seconds

if __name__ == "__main__":
    main()
