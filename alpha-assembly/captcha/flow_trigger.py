# are we on captcha state ?
from captcha.setup.challenge_finder import find_challenge
from captcha.process_captcha import process_frame_for_heatmap, process_frame_for_ocr
import cv2
import numpy as np
from captcha.setup.find_pokemon_position import PokemonFinder
from captcha.solve.pokeball_detector import PokeballDetector
import os
# turn this into a class
class FlowTrigger:
    def __init__(self):
        self.pokeball_detector = PokeballDetector(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/templates/pokeball_template2.png"))
        self.captcha_state = False
        self.is_captcha_solving_state = False
        self.captcha_challenge_position = None

    def is_captcha_state(self, frame):
        processed_frame = frame
        template = cv2.imread('assets/templates/captcha_template.png', cv2.IMREAD_GRAYSCALE)
        
        if template is None:
            print("Error: Could not load template")
            return False
            
        template_h, template_w = template.shape
        frame_h, frame_w = processed_frame.shape
        
        # Calculate maximum possible scale
        max_scale = min(frame_w / template_w, frame_h / template_h)
        
        # Generate scales that make sense for the image sizes
        base_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        scales = [s for s in base_scales if s <= max_scale]
        
        if not scales:
            print("No valid scales found - template might be too large")
            return False
        
        matches = []
        
        for scale in scales:
            width = int(template_w * scale)
            height = int(template_h * scale)
            
            resized = cv2.resize(template, (width, height))
            
            result = cv2.matchTemplate(processed_frame, resized, cv2.TM_CCOEFF_NORMED)
            
            threshold = 0.6  # Lower initial threshold to catch potential matches
            loc = np.where(result >= threshold)
            
            for pt in zip(*loc[::-1]):
                x = int(pt[0] + width/2)
                y = int(pt[1] + height/2)
                r = int(width/2)
                score = result[pt[1], pt[0]]
                matches.append((x, y, r, score, scale))
        
        # Sort matches by score
        matches.sort(key=lambda x: x[3], reverse=True)
        
        # Filter overlapping matches
        filtered_matches = []
        for m in matches:
            x1, y1, r1, score, scale = m
            if not any(np.sqrt((x1-x2)**2 + (y1-y2)**2) < max(r1, r2) 
                    for (x2, y2, r2, _, _) in filtered_matches):
                filtered_matches.append(m)
        
        # Draw debug visualization
        debug_frame = frame.copy()
        for x, y, r, score, scale in filtered_matches[:3]:  # Show top 3 matches
            cv2.circle(debug_frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Score: {score:.2f} Scale: {scale:.1f}", 
                    (x-r, y-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print best match info
        if filtered_matches:
            best_match = filtered_matches[0]
            print(f"Best match - Score: {best_match[3]:.3f}, Scale: {best_match[4]:.2f}")
        
            return len(filtered_matches) > 0 and filtered_matches[0][3] > 0.65

    def verify_and_execute_captcha_state(self, frame):
        processed_frame = process_frame_for_heatmap(frame)

        if not self.captcha_state and self.is_captcha_state(processed_frame):
            print("Captcha state detected")
            processed_frame_ocr = process_frame_for_ocr(frame)
            processed_frame_ocr = cv2.resize(processed_frame_ocr, (0, 0), fx=0.4, fy=0.4)
            #save this frame
            cv2.imwrite('debug/processed_frame.png', processed_frame_ocr)
            challenge = find_challenge(processed_frame_ocr)
            finder = PokemonFinder(challenge)
            pokemon_position = finder.find_on_screen(frame)
            print("pokemon_position")
            print(pokemon_position)
            self.captcha_challenge_position = pokemon_position
            self.captcha_state = True
            print("finished captcha setup stage")
            # convert processed frame to rgb
            
            # TODO: execute the captcha state
            # challenge = find_challenge(frame)

        if self.captcha_state:
            pokeball_positions = self.pokeball_detector.find_pokeballs(frame)
            if(len(pokeball_positions) >= 2):
                print("found 3 or more pokeballs for this frame")
                challenge_x, challenge_y = self.captcha_challenge_position
                
                # Calculate center points for each pokeball and find the closest
                pokeball_centers = [(x, y) for x, y, _, _ in pokeball_positions]
                closest_center = min(pokeball_centers, key=lambda center: np.sqrt((center[0]-challenge_x)**2 + (center[1]-challenge_y)**2))
               
                # Find the original pokeball data that matches this center
                closest_pokeball = next(ball for ball in pokeball_positions if ball[0] == closest_center[0] and ball[1] == closest_center[1])
                
                print("closest_pokeball")
                print(closest_pokeball)
                # always use the center of the pokeball as the challenge position
                self.captcha_challenge_position = (closest_pokeball[0], closest_pokeball[1])
                
                # draw a circle on the screen based on the closest pokeball
                frame_copy = frame.copy()
                cv2.circle(frame_copy, (closest_pokeball[0], closest_pokeball[1]), closest_pokeball[2], (0, 0, 255), 2)
                cv2.circle(frame_copy, (closest_pokeball[0], closest_pokeball[1]), 10, (0, 255, 0), 2)
                cv2.imshow("closest_pokeball", frame_copy)
                # Update window without blocking (1ms delay)
                cv2.waitKey(1)
            print("captcha state")
