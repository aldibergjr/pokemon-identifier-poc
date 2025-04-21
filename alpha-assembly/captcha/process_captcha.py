import cv2
import numpy as np

def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2.4, img.shape[0] / 4
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def process_frame_for_heatmap(frame):
    """
    Process a frame to isolate yellow text
    Args:
        frame: numpy array in BGR format
    Returns:
        processed frame with only yellow text visible as a single channel image
    """
    # Save original frame for debugging
    cv2.imwrite('debug/original.png', frame)
    
    # Crop to be roughly in the middle of the screen
    frame = crop_img(frame, 0.4)
    cv2.imwrite('debug/cropped.png', frame)
    
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imwrite('debug/hsv.png', hsv)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create mask for yellow text
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imwrite('debug/initial_mask.png', mask)

    # Apply some morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # save final mask
    cv2.imwrite('debug/final_mask.png', mask)
    return mask  # Return the single-channel mask directly


def process_frame_for_ocr(frame):
    """
    Process a frame to isolate yellow text
    Args:
        frame: numpy array in BGR format
    Returns:
        processed frame with only yellow text visible as a single channel image
    """
    return crop_img(frame, 0.4)


# Example usage:
# frame = your_numpy_array  # Your input frame
# processed_frame = process_frame(frame) 