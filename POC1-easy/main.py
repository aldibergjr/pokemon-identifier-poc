import cv2
import numpy as np
import pyautogui
import time
import os
from pokeball_detector import PokeballDetector
from pokemon_detector import PokemonDetector

# Create directories if they don't exist
os.makedirs('debug', exist_ok=True)
os.makedirs('debug/regions', exist_ok=True)
os.makedirs('debug/matching', exist_ok=True)
os.makedirs('assets/text_templates', exist_ok=True)

# Parameters
capture_region = (0, 0, 1920, 1080)

def main():
    # Initialize detectors
    pokeball_detector = PokeballDetector('assets/templates/pokeball_template.png')
    pokemon_detector = PokemonDetector()
    
    print("Looking for pokemon and pokeballs... Press Ctrl+C to stop")
    print("Debug images will be saved in the debug folders")
#print(f"Pokemon detection is {'enabled' if pokemon_detector.templates else 'disabled (no templates found)'}")

    try:
        while True:
            # Capture screen
            screenshot = pyautogui.screenshot(region=capture_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save full screenshot for debugging
            cv2.imwrite('debug/full_screen.png', frame)
            
            # Clear previous debug regions
            for f in os.listdir('debug/regions'):
                if f.endswith('.png'):
                    try:
                        os.remove(os.path.join('debug/regions', f))
                    except:
                        pass
            
            # Find pokeballs
            pokeball_matches = pokeball_detector.find_pokeballs(frame)
            
            # Find pokemon
            pokemon_matches = pokemon_detector.find_pokemon(frame)
            print("tá vivo")
            # Clear the console and print results
            if os.name == 'nt':  # for Windows
                os.system('cls')
            else:  # for Unix/Linux/MacOS
                os.system('clear')
            
            print("\n=== Detection Results ===")
            
            # Print pokemon matches
            if pokemon_matches:
                print(f"\nFound {len(pokemon_matches)} pokemon:")
                for name, x, y, w, h, score in pokemon_matches:
                    print(f"Pokemon: {name}, Position: ({x}, {y}), Size: {w}x{h}, Confidence: {score:.2f}")
            else:
                print("No pokemon detected")
            
            # Print pokeball matches
            if pokeball_matches:
                top_matches = sorted(pokeball_matches, key=lambda x: x[3], reverse=True)[:3]
                print(f"\nFound {len(top_matches)} pokeballs:")
                for idx, (x, y, r, score) in enumerate(top_matches):
                    print(f"Pokeball {idx+1}: Position: ({x}, {y}), Radius: {r}, Confidence: {score:.2f}")
                
                # Check if we found a wild Pokemon (3 pokeballs and pokemon text)
                if len(top_matches) == 3 and all(score > 0.6 for _, _, _, score in top_matches):
                    if pokemon_matches:
                        print("\n!!! WILD POKEMON DETECTED !!!")
                        for name, x, y, w, h, score in pokemon_matches:
                            print(f"Pokemon: {name} (Confidence: {score:.2f})")
                        
                        # Save this successful detection
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(f'debug/detection_{timestamp}.png', frame)
                        
                        print("Found all 3 pokeballs with good confidence!")
                        user_input = input("\nPress Enter to continue or 'q' to quit: ")
                        if user_input.lower() == 'q':
                            break
            else:
                print("No pokeballs detected")
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping detection...")

if __name__ == "__main__":
    main()

