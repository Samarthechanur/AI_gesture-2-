import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import pyttsx3
from datetime import datetime

# --- Text To Speech ---
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Load Model & Labels ---
MODEL_PATH = "model/hand_gesture_model.h5"
LABELS_PATH = "labels.txt"
DATASET_ROOT = "My_dataset"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# --- Canvas & Settings ---
canvas_width, canvas_height = 640, 480
side_pad, top_pad, bottom_pad = 30, 60, 70
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
drawing_mode = False
prev_point = None
prediction_text = ""
expression = ""

PRIMARY_BG = (245, 245, 245)
CARD_BG = (230, 240, 250)
BORDER_COLOR = (150, 100, 220)
CANVAS_BG = (30, 34, 40)
CAMERA_CARD = (200, 215, 235)

instructions = [
    "S:start/stop drawing | Enter: finalize | C: clear canvas",
    "Keys 1:Letter/Number | 2:Vocabulary | 3:Calculator | Q: Quit"
]

mode_names = ["Letter/Number Recognition", "Vocabulary Gesture", "Calculator"]
current_mode = 0  # Start in Letter/Number mode

# --- Utility ---
def draw_card(img, x, y, w, h, color, border=BORDER_COLOR):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), border, 2)
    return cv2.addWeighted(overlay, 0.97, img, 0.03, 0)

def save_letter_image(img, label_char):
    if label_char.isdigit():
        folder_path = os.path.join(DATASET_ROOT, "Numbers", label_char)
    else:
        folder_path = os.path.join(DATASET_ROOT, "Letters", label_char.upper())
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"{label_char}_{timestamp}.png"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, img)
    print(f"[INFO] Saved '{label_char}' to {filepath}")

def interpolate_points(p1, p2):
    points = []
    p1_arr, p2_arr = np.array(p1), np.array(p2)
    dist = int(np.linalg.norm(p2_arr - p1_arr))
    step = 2
    if dist == 0:
        return [tuple(p2_arr)]
    for i in range(0, dist + 1, step):
        pt = tuple((p1_arr + (p2_arr - p1_arr) * i / dist).astype(int))
        points.append(pt)
    return points

def segment_and_preprocess_letters(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    preprocessed = []
    for box in sorted_boxes:
        x, y, w, h = box
        if w < 12 or h < 12:
            continue
        letter_img = thresh[y:y + h, x:x + w]
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        x_off, y_off = (size - w) // 2, (size - h) // 2
        square[y_off:y_off + h, x_off:x_off + w] = letter_img
        resized = cv2.resize(square, (28, 28))
        norm_img = resized.astype("float32") / 255.0
        norm_img = np.expand_dims(norm_img, axis=-1)
        norm_img = np.expand_dims(norm_img, axis=0)
        preprocessed.append((norm_img, resized))
    return preprocessed

def vocabulary_gesture_detect(hand_landmarks):
    if hand_landmarks is None:
        return None
    fingers_open = []
    tips = [4, 8, 12, 16, 20]
    for tip in tips:
        fingertip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        fingers_open.append(fingertip_y < pip_y)
    if fingers_open == [True, True, True, True, True]:
        return "Hello"
    elif fingers_open == [False, True, False, False, False]:
        return "Yes"
    elif fingers_open == [False, False, False, False, False]:
        return "No"
    else:
        return None

def calculator_eval(expr):
    try:
        allowed_chars = "0123456789+-*/. "
        if any(c not in allowed_chars for c in expr):
            return "Invalid Expression"
        result = eval(expr)
        return str(result)
    except Exception:
        return "Error"

cap = cv2.VideoCapture(0)
prev_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape
    index_finger_tip = None
    detected_phrase = None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if len(handLms.landmark) < 9:
                continue
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)
            if 0 < x < w and 0 < y < h:
                index_finger_tip = (x, y)

            if current_mode == 1:
                detected_phrase = vocabulary_gesture_detect(handLms)

    full_w, full_h = 2 * canvas_width + 2 * side_pad, canvas_height + top_pad + bottom_pad
    display = np.full((full_h, full_w, 3), PRIMARY_BG, dtype=np.uint8)
    display = draw_card(display, side_pad, top_pad, canvas_width, canvas_height, CAMERA_CARD)
    cam_resized = cv2.resize(frame, (canvas_width, canvas_height))
    display[top_pad:top_pad + canvas_height, side_pad:side_pad + canvas_width] = cam_resized
    display = draw_card(display, side_pad + canvas_width, top_pad, canvas_width, canvas_height, CANVAS_BG)
    canvas_overlay = cv2.addWeighted(canvas, 0.7, np.full_like(canvas, CANVAS_BG), 0.3, 0)
    display[top_pad:top_pad + canvas_height, side_pad + canvas_width:side_pad + 2 * canvas_width] = canvas_overlay

    cv2.rectangle(display, (0, 0), (full_w, top_pad - 8), CARD_BG, -1)
    cv2.putText(display, f"Mode: {mode_names[current_mode]}", (side_pad + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 0, 120), 3)

    cv2.rectangle(display, (0, full_h - bottom_pad), (full_w, full_h), CARD_BG, -1)
    for i, instr in enumerate(instructions):
        cv2.putText(display, instr, (side_pad + 10, full_h - bottom_pad + 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 100, 160), 2)

    if prediction_text:
        cv2.putText(display, prediction_text, (side_pad + canvas_width + 20, top_pad + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 255, 115), 4)

    if current_mode == 1 and detected_phrase:
        cv2.putText(display, f"Detected: {detected_phrase}", (side_pad + canvas_width + 20, top_pad + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 190, 0), 4)

    cv2.imshow("AI Gesture Studio - MultiMode", display)
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
    elif key in [ord('1'), ord('2'), ord('3')]:
        current_mode = key - ord('1')
        prediction_text = ""
        expression = ""
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        drawing_mode = False
        print(f"[INFO] Switched to mode: {mode_names[current_mode]}")
    elif key == 13:
        if current_mode == 0:
            print("[INFO] Predicting and saving letters/numbers...")
            letters = segment_and_preprocess_letters(canvas)
            predicted_word = ""
            for pt_img, raw_img in letters:
                preds = model.predict(pt_img)
                pred_index = np.argmax(preds)
                pred_label = labels[pred_index]
                predicted_word += pred_label
                save_letter_image(raw_img, pred_label)
            if predicted_word:
                prediction_text = f"Word: {predicted_word}"
                print(f"[RESULT] Predicted word: {predicted_word}")
                speak_text(predicted_word)
            else:
                prediction_text = "No letters detected"
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        elif current_mode == 1:
            if detected_phrase:
                speak_text(detected_phrase)
                prediction_text = f"Vocabulary Word: {detected_phrase}"
            else:
                prediction_text = "No vocabulary gesture detected"
        elif current_mode == 2:
            print("[INFO] Recognizing expression...")
            letters = segment_and_preprocess_letters(canvas)
            expr = ""
            for pt_img, _ in letters:
                preds = model.predict(pt_img)
                pred_index = np.argmax(preds)
                pred_label = labels[pred_index]
                expr += pred_label
            prediction_text = f"Expr: {expr}"
            print(f"[Calculator] Expression: {expr}")
            try:
                result = eval(expr)
            except:
                result = "Error"
            print(f"[Calculator] Result: {result}")
            speak_text(str(result))
            prediction_text += f" | Result: {result}"
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Drawing logic
    if drawing_mode and current_mode in [0, 2]:  # drawing allowed in Letter/Number and Calculator modes
        if index_finger_tip is not None:
            if prev_point is not None:
                points = interpolate_points(prev_point, index_finger_tip)
                for pt in points:
                    cv2.circle(canvas, pt, 8, (255, 255, 255), -1)
            prev_point = index_finger_tip
        else:
            prev_point = None
    else:
        prev_point = None

cap.release()
cv2.destroyAllWindows()
