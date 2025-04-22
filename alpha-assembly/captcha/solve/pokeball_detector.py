import cv2
import numpy as np

class PokeballDetector:
    def __init__(self, template_path):
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise Exception(f"Error: Could not load template from {template_path}")
            
    def find_pokeballs(self, screen):
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        template_h, template_w = template_gray.shape
        
        matches = []
        scales = [0.35, 0.4, 0.45, 0.5, 0.6]
        
        for scale in scales:
            width = int(template_w * scale)
            height = int(template_h * scale)
            resized = cv2.resize(template_gray, (width, height))
            
            result = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            
            threshold = 0.7
            loc = np.where(result >= threshold)
            
            for pt in zip(*loc[::-1]):
                x = int(pt[0] + width/2)
                y = int(pt[1] + height/2)
                r = int(width/2)
                score = result[pt[1], pt[0]]
                matches.append((x, y, r, score))
        
        matches.sort(key=lambda x: x[3], reverse=True)

        filtered_matches = []
        
        for m in matches:
            x1, y1, r1, _ = m
            if not any(np.sqrt((x1-x2)**2 + (y1-y2)**2) < max(r1, r2) 
                      for (x2, y2, r2, _) in filtered_matches):
                filtered_matches.append(m)
        
        # draw a circle on the screen based on the matches
        # if(len(filtered_matches) >= 3):
        #     for x1, y1, r1, _ in filtered_matches:
        #         cv2.circle(screen, (x1, y1), r1, (0, 0, 255), 2)
        #     cv2.imshow("screen", screen)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return filtered_matches