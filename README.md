# AI Gesture Studio – Air-Writing Recognition

AI Gesture Studio is a real-time **air-writing recognition system** that uses a webcam and MediaPipe hand-tracking to capture finger movements in free air, then predicts drawn **digits (0‑9)** and **uppercase letters (A‑Z)** using a deep learning CNN model.

## 📌 Features
- **Real-time Hand Tracking** (MediaPipe)
- **Draw Letters or Numbers in Air** and recognize them instantly
- **Single Character & Full Word Prediction**
- **Accurate CNN Model** (trained on EMNIST Balanced + augmentation)
- **Polished Interface** with instructions and controls
- **Saves Predictions** for retraining or auditing

## 🎮 Controls
| Key | Action |
|-----|--------|
| **S** | Start/Stop Drawing |
| **E** | Predict and Add Current Character to Word |
| **C** | Clear Canvas |
| **Enter** | Finalize Word and Display Prediction |
| **Q** | Quit |

## 🛠 Technologies Used
- Python 3
- OpenCV
- MediaPipe
- TensorFlow/Keras
- NumPy

## 🚀 Usage
1. Clone this repository:
    ```
    git clone https://github.com/<your-username>/AI_Gesture_Studio.git
    ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the app:
    ```
    python hand_gesture_recognition.py
    ```

## 📂 Project Structure
   AI_Gesture_Studio/
│-- model/hand_gesture_model.h5 # Trained CNN model
│-- labels.txt # Label list (0-9, A-Z)
│-- hand_gesture_recognition.py # Main app
│-- gesture_utils.py # Utilities for preprocessing
│-- train_model.py / train_model_augmented.py
│-- README.md

## 👨‍💻 Author
Samarth H – [GitHub Profile](https://github.com/Samarthechanur)

