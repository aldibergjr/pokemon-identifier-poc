import cv2
import numpy as np

def crop_img(img, scale=1.0, centerx = 2.4, centery = 4):
    center_x, center_y = img.shape[1] / centerx, img.shape[0] / centery
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
    Process a frame to isolate text with its outline and remove noise
    Args:
        frame: numpy array in BGR format
    Returns:
        processed frame with clean text and outline
    """
    # Save original frame for debugging
    cv2.imwrite('debug/ocr_original.png', frame)
    
    # Crop to focus on text area
    frame = crop_img(frame, 0.4)
    cv2.imwrite('debug/ocr_cropped.png', frame)
    
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define strict yellow color ranges for the bright text
    lower_yellow = np.array([20, 150, 180])  # More saturated and brighter yellows
    upper_yellow = np.array([35, 255, 255])
    
    # Create yellow mask for the text
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Define a slightly more permissive range for the outline
    lower_yellow_outline = np.array([20, 100, 100])
    upper_yellow_outline = np.array([35, 255, 255])
    outline_mask = cv2.inRange(hsv, lower_yellow_outline, upper_yellow_outline)
    
    # Combine text and outline
    kernel = np.ones((2,2), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
    combined_mask = cv2.bitwise_or(yellow_mask, outline_mask)
    
    # Clean up noise
    # Use a slightly larger kernel for noise removal
    kernel_clean = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean)
    
    # Remove small isolated components more aggressively
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Increased threshold to remove more noise
            cv2.drawContours(mask, [contour], -1, 0, -1)
    
    # Final cleanup
    mask = cv2.medianBlur(mask, 3)  # Remove salt-and-pepper noise
    
    cv2.imwrite('debug/ocr_final_mask.png', mask)
    
    # Create final result
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imwrite('debug/ocr_final_result.png', result)
    
    # # Convert black background to white for better OCR
    # result[mask == 0] = [255, 255, 255]
    
    return result


# Example usage:
# frame = your_numpy_array  # Your input frame
# processed_frame = process_frame(frame) 