import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os
import pyttsx3

# ========= TEXT-TO-SPEECH SETUP ==========
def speak_text(text):
    """Speaks the given text aloud using pyttsx3 text-to-speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ========== MODEL & UTILS SETUP ==========
MODEL_PATH = "model/hand_gesture_model.h5"
LABELS_PATH = "labels.txt"
SAVE_DIR = "saved"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# MediaPipe hand tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

canvas_width, canvas_height = 640, 480
side_pad = 30
top_pad = 60
bottom_pad = 70

canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

drawing_mode = False
prev_point = None
prediction_text = ""

word_buffer = []
canvas_clear_flag = False

PRIMARY_BG = (245, 245, 245)
CARD_BG = (230, 240, 250)
BORDER_COLOR = (150, 100, 220)
CANVAS_BG = (30, 34, 40)
CAMERA_CARD = (200, 215, 235)

instructions = [
    "S: Start/Stop Drawing  |  E: Add Letter/Number  |  C: Clear Canvas",
    "Enter: Finalize Word & Show Prediction  |  Q: Quit"
]

def draw_card(img, x, y, w, h, color, border=BORDER_COLOR):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), border, 2)
    return cv2.addWeighted(overlay, 0.97, img, 0.03, 0)

def save_drawing(img, pred_label, cur_word):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    word_folder = os.path.join(SAVE_DIR, f"word_{cur_word or 'untitled'}_{timestamp}")
    os.makedirs(word_folder, exist_ok=True)
    filename = f"{pred_label}_{timestamp}.png"
    filepath = os.path.join(word_folder, filename)
    cv2.imwrite(filepath, img)
    print(f"[INFO] Saved drawing as {filepath}")

# ==== FIXED INTERPOLATE POINTS FUNCTION ====
def interpolate_points(p1, p2):
    points = []
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)
    dist = int(np.linalg.norm(p2_arr - p1_arr))
    if dist == 0:
        return [tuple(p2_arr)]
    for i in range(dist):
        pt = tuple((p1_arr + (p2_arr - p1_arr) * i / dist).astype(int))
        points.append(pt)
    return points

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

# ================= MAIN LOOP ==================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape
    index_finger_tip = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)
            index_finger_tip = (x, y)
            cv2.circle(frame, (x, y), 10, (0, 153, 255), cv2.FILLED)

    if drawing_mode and index_finger_tip is not None:
        if prev_point is None:
            prev_point = index_finger_tip
        points = interpolate_points(prev_point, index_finger_tip)
        for pt in points:
            cv2.circle(canvas, pt, 8, (255, 255, 255), -1)
        prev_point = index_finger_tip
    else:
        prev_point = None

    full_w, full_h = 2 * canvas_width + 2 * side_pad, canvas_height + top_pad + bottom_pad
    display = np.full((full_h, full_w, 3), PRIMARY_BG, dtype=np.uint8)

    display = draw_card(display, side_pad, top_pad, canvas_width, canvas_height, CAMERA_CARD)
    cam_resized = cv2.resize(frame, (canvas_width, canvas_height))
    display[top_pad:top_pad + canvas_height, side_pad:side_pad + canvas_width] = cam_resized

    display = draw_card(display, side_pad + canvas_width, top_pad, canvas_width, canvas_height, CANVAS_BG)
    canvas_overlay = cv2.addWeighted(canvas, 0.7, np.full_like(canvas, CANVAS_BG), 0.3, 0)
    display[top_pad:top_pad + canvas_height, side_pad + canvas_width:side_pad + 2 * canvas_width] = canvas_overlay

    cv2.rectangle(display, (0, 0), (full_w, top_pad - 8), CARD_BG, -1)
    cv2.putText(display, "AI Gesture Studio - Air-Written Word Recognition",
                (full_w // 2 - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 40, 200), 3, cv2.LINE_AA)

    cv2.rectangle(display, (0, full_h - bottom_pad), (full_w, full_h), CARD_BG, -1)
    for i, instr in enumerate(instructions):
        cv2.putText(display, instr, (side_pad + 10, full_h - bottom_pad + 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 100, 160), 2)

    if word_buffer:
        cv2.putText(display, f"Current Word ({len(word_buffer)} letters): {''.join(word_buffer)}",
                    (side_pad + 15, top_pad + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (40, 100, 240), 4, cv2.LINE_AA)

    if prediction_text:
        cv2.putText(display, prediction_text,
                    (side_pad + canvas_width + 20, top_pad + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90, 255, 115), 4)

    cv2.imshow("AI Gesture Studio", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        drawing_mode = not drawing_mode
        print("[INFO] Drawing started." if drawing_mode else "[INFO] Drawing stopped.")
    elif key == ord('c'):
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        prediction_text = "Canvas cleared!"
        print("[INFO] Canvas cleared.")
    elif key == ord('e') and not drawing_mode:
        roi_input = preprocess_canvas(canvas)
        if roi_input is not None:
            prediction = model.predict(roi_input)
            pred_index = np.argmax(prediction)
            pred_label = labels[pred_index]
            word_buffer.append(pred_label)
            prediction_text = f"Added: {pred_label}"
            img_to_save = (roi_input[0] * 255).astype("uint8")
            if img_to_save.shape[-1] == 1:
                img_to_save = img_to_save.squeeze(-1)
            save_drawing(img_to_save, pred_label, ''.join(word_buffer))
            canvas_clear_flag = True

            # === TEXT TO SPEECH ===
            speak_text(f"Added letter {pred_label}")
        else:
            prediction_text = "No drawing to predict"
            print("[INFO] Nothing drawn.")
    elif key == 13:  # Enter key
        if word_buffer:
            word_str = "".join(word_buffer)
            prediction_text = "Full Word: " + word_str
            print(f"[RESULT] Full Word prediction: {word_str}")
            # ============== KEY CHANGE HERE: Pronounce as a word ==============
            speak_text(word_str)  # Only the word (WAR), not letter by letter!
            word_buffer = []
        else:
            prediction_text = "No letters drawn"

    if canvas_clear_flag:
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas_clear_flag = False

cap.release()
cv2.destroyAllWindows()
