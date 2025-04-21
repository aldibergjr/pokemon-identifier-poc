import cv2
import numpy as np

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2.4, img.shape[0] / 4
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def process_frame(frame):
    """
    Process a frame to isolate yellow text
    Args:
        frame: numpy array in BGR format
    Returns:
        processed frame with only yellow text visible
    """
    # Crop to be roughly in the middle of the screen
    frame = crop_img(frame, 0.4)
    
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create mask for yellow text
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply some morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Create the final image
    result = frame.copy()
    result[mask == 0] = [0, 0, 0]  # Set non-text pixels to black

    return result

# Example usage with image file
if __name__ == "__main__":
    frame = cv2.imread('image.png')
    processed_frame = process_frame(frame)
    cv2.imwrite('processed_image.png', processed_frame)
