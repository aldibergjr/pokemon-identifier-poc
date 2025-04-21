import cv2
import numpy as np
import os
import easyocr
from PIL import Image
import tempfile

class PokemonDetector:
    def __init__(self, font_path='assets/fonts/Ketchum.otf'):
        """Initialize the Pokemon Detector with the game's font."""
        self.font_path = font_path
        # Initialize EasyOCR with Portuguese and English
        self.reader = easyocr.Reader(['pt', 'en'], gpu=False)
        # Known text patterns
        self.text_patterns = {
            "question": "Onde está",
            "pokemon": "Dragonair"
        }

    def preprocess_image(self, screen):
        """Preprocess the image to isolate yellow text with black outline."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV (more permissive range)
        lower_yellow = np.array([15, 70, 150])
        upper_yellow = np.array([45, 255, 255])
        
        # Create mask for yellow text
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Create a mask for the black outline
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(yellow_mask, black_mask)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours to get text regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask with just the largest contours (text regions)
        text_mask = np.zeros_like(combined_mask)
        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Take the top 2-3 largest contours (likely the text)
            for i in range(min(3, len(contours))):
                cv2.drawContours(text_mask, [contours[i]], -1, (255), -1)
        
        # Apply the refined mask
        result = cv2.bitwise_and(screen, screen, mask=text_mask)
        
        # Save intermediate results for debugging
        cv2.imwrite('debug/yellow_mask.png', yellow_mask)
        cv2.imwrite('debug/black_mask.png', black_mask)
        cv2.imwrite('debug/text_mask.png', text_mask)
        cv2.imwrite('debug/combined_mask.png', combined_mask)
        cv2.imwrite('debug/result.png', result)
        
        return result

    def extract_text(self, screen):
        """Extract Pokemon name from the screen image using EasyOCR."""
        # Preprocess the image
        processed = self.preprocess_image(screen)
        
        try:
            # Use EasyOCR to detect text
            results = self.reader.readtext(processed)
            
            # Process results
            texts = []
            for (bbox, text, prob) in results:
                text = text.strip()
                if text:
                    print(f"Detected text: {text} (confidence: {prob:.2f})")
                    texts.append(text)
            
            # Try to find matches with known patterns
            matched_texts = []
            for text in texts:
                text_lower = text.lower()
                for pattern_name, pattern in self.text_patterns.items():
                    if pattern.lower() in text_lower:
                        matched_texts.append(text)
                        break
            
            # If we found pattern matches, use those; otherwise use all detected text
            final_texts = matched_texts if matched_texts else texts
            
            # Combine all detected text
            return ' '.join(final_texts)
            
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return ''

    def find_pokemon(self, screen):
        """Find Pokemon names in the screen image using OCR."""
        text = self.extract_text(screen)
        if text:
            # Return in a similar format to the original template matching
            height, width = screen.shape[:2]
            return [(text, 0, 0, width, height, 1.0)]
        return [] 