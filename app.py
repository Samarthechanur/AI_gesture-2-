import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import os

st.set_page_config(page_title="AI Gesture Studio", page_icon="🤖", layout="wide")

MODEL_PATH = 'model/model/hand_gesture_model.h5'
LABELS_PATH = 'model/labels.txt'
SAVE_DIR = 'saved'
os.makedirs(SAVE_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def preprocess_canvas(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    preprocessed = []
    for x,y,w,h in bounding_boxes:
        if w < 12 or h < 12:
            continue
        letter_img = thresh[y:y+h, x:x+w]
        size = max(w,h)
        square = np.zeros((size,size), dtype=np.uint8)
        x_off, y_off = (size-w)//2, (size-h)//2
        square[y_off:y_off+h, x_off:x_off+w] = letter_img
        resized = cv2.resize(square, (28,28))
        norm_img = resized.astype("float32") / 255.
        norm_img = np.expand_dims(norm_img, axis=-1)
        norm_img = np.expand_dims(norm_img, axis=0)
        preprocessed.append((norm_img, resized))
    return preprocessed

# Initialize session state
if "canvas" not in st.session_state:
    st.session_state.canvas = np.zeros((480,640,3), dtype=np.uint8)
if "drawing_mode" not in st.session_state:
    st.session_state.drawing_mode = False
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "prev_point" not in st.session_state:
    st.session_state.prev_point = None
if "prediction_text" not in st.session_state:
    st.session_state.prediction_text = ""
if "word_buffer" not in st.session_state:
    st.session_state.word_buffer = []

st.title("AI Gesture Studio – Air-Writing Recognition")

st.markdown("""
### Instructions:
- Click **Start Camera** to activate webcam.
- Toggle drawing using **Start/Stop Drawing (S)**.
- Draw letters with your index finger.
- Click **Predict Word (Enter)** to run prediction.
- Use **Add Space** to add spaces in the sentence.
- Click **Clear Letter** to undo last letter.
- Click **Clear Canvas** to reset drawing and stop camera.
""")

cols = st.columns(6)
cam_btn = cols[0].button("Start Camera" if not st.session_state.camera_on else "Stop Camera")
draw_btn = cols[1].button("Start/Stop Drawing (S)")
predict_btn = cols[2].button("Predict Word (Enter)")
space_btn = cols[3].button("Add Space")
clear_letter_btn = cols[4].button("Clear Letter")
clear_btn = cols[5].button("Clear Canvas")

if cam_btn:
    st.session_state.camera_on = not st.session_state.camera_on
    if not st.session_state.camera_on:
        st.session_state.drawing_mode = False
        st.session_state.prev_point = None

if draw_btn:
    if st.session_state.camera_on:
        st.session_state.drawing_mode = not st.session_state.drawing_mode
    else:
        st.warning("Please start the camera first.")

if space_btn:
    st.session_state.word_buffer.append(" ")

if clear_letter_btn and st.session_state.word_buffer:
    st.session_state.word_buffer.pop()

if clear_btn:
    st.session_state.canvas = np.zeros((480,640,3), dtype=np.uint8)
    st.session_state.prediction_text = ""
    st.session_state.word_buffer = []
    st.session_state.camera_on = False
    st.session_state.drawing_mode = False
    st.session_state.prev_point = None

if predict_btn:
    letters = preprocess_canvas(st.session_state.canvas)
    predicted_word = ""
    for img_input, _ in letters:
        preds = model.predict(img_input)
        pred_index = preds.argmax()
        pred_char = labels[pred_index]
        predicted_word += pred_char
    if predicted_word:
        st.session_state.prediction_text = predicted_word
        st.session_state.word_buffer.extend(list(predicted_word))
        speak_text(predicted_word)
    else:
        st.warning("No letters detected. Please draw clearly.")
    st.session_state.canvas = np.zeros((480,640,3), dtype=np.uint8)

frame_placeholder = st.empty()
canvas_placeholder = st.empty()

if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
        st.session_state.camera_on = False
    else:
        col_cam, col_canvas = st.columns(2)
        with col_cam:
            frame_placeholder = st.empty()
        with col_canvas:
            canvas_placeholder = st.empty()

        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video frame.")
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            h, w, _ = frame.shape
            index_finger_pos = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark[8]
                    x, y = int(lm.x * w), int(lm.y * h)
                    index_finger_pos = (x, y)

            if st.session_state.drawing_mode and index_finger_pos and st.session_state.prev_point:
                cv2.line(st.session_state.canvas, st.session_state.prev_point, index_finger_pos, (255, 255, 255), 8)

            st.session_state.prev_point = index_finger_pos if st.session_state.drawing_mode else None

            canvas_disp = cv2.addWeighted(st.session_state.canvas, 0.7,
                                         np.full_like(st.session_state.canvas, (30, 34, 40)), 0.3, 0)

            frame_placeholder.image(frame, channels="BGR", caption="Webcam Feed")
            canvas_placeholder.image(canvas_disp, caption="Drawing Canvas")

            if cv2.waitKey(1) & 0xFF == 27:
                st.session_state.camera_on = False
                st.session_state.drawing_mode = False
                break
        cap.release()

if st.session_state.word_buffer:
    st.markdown(f"### Current Sentence: **{''.join(st.session_state.word_buffer)}**")
if st.session_state.prediction_text:
    st.markdown(f"### Last Predicted Word: **{st.session_state.prediction_text}**")
