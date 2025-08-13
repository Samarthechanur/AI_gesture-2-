import cv2
import numpy as np

def preprocess_canvas(canvas):
    # Convert the canvas to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # Threshold to get binary image
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # Get bounding box of largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Crop and resize to 28x28 as expected by model
    roi = thresh[y:y+h, x:x+w]
    roi = cv2.resize(roi, (28, 28))
    # Normalize and add batch/channel dimensions
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi
