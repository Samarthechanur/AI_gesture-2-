# AI Gesture Studio – Air‑Written Hand Gesture Recognition with Text‑to‑Speech

A computer vision project that enables users to write letters and numbers in the air using hand gestures, recognizes them with a machine learning model, and converts them to text using offline speech synthesis.


##  About the Project

**AI Gesture Studio** is an assistive and innovative application that recognizes air-written hand gestures using a webcam and converts them into text with offline text-to-speech functionality. It’s designed for  People with speech disabilities or difficulties, enabling them to communicate more easily through air-written gestures and offline text-to-speech and also researchers and students exploring gesture recognition, assistive technologies, and human-computer interaction. 

## Features

- Air-writing recognition using finger movements.
- Prediction of letters (A–Z) and digits (0–9).
- Offline text-to-speech using `pyttsx3`.
- User-controlled interaction to avoid accidental triggers.
- Smooth interpolation for clear handwriting recognition.
- Real-time visualization with OpenCV.

## Authors

- [@Pawan P Acharya](https://github.com/PAWANPACHARYA)
- [@Pranam](https://github.com/sappranam)
- [@Samarth H](https://github.com/Samarthechanur)
- [@Sankalp Poojary](https://github.com/sankaalp)

## Acknowledgements

We extend our gratitude to:

- Our mentors and faculty for their guidance.
- The open-source community behind TensorFlow/Keras, MediaPipe Hands, OpenCV, and Python libraries like `pyttsx3` that enabled offline text-to-speech.
- All contributors and developers whose resources have made this project possible.

## Tech Stack

- **TensorFlow/Keras** – Model inference for gesture recognition.
- **MediaPipe Hands** – Real-time hand landmark detection.
- **OpenCV** – Webcam capture, drawing, and image segmentation.
- **pyttsx3** – Offline text-to-speech engine.
- **Python** – Core programming language.
- **PyInstaller** – Building standalone executables for Windows.

## Installation

### Requirements

- **Operating System**: Windows 10/11 (x64). Linux/macOS may work with compatible packages.
- **Python Version**: 3.10 recommended.
- **Hardware**:
  - Any modern x64 CPU with at least 4 GB RAM (8 GB recommended).
  - Webcam and speakers.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Samarthechanur/AI_gesture-2-
   ```

2. Navigate to the project directory:
   ```bash
   cd AI_gesture-2--main/AI_gesture-2--main
   ```

3. Create or activate a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##  Usage

Start the application by running:

```bash
python hand_gesture_recognition.py
```

### Runtime Shortcuts

- **S** – Start/Stop air-writing.
- **E** – Segment and predict the written text.
- **Enter** – Speak the current buffer.
- **Space** – Insert a space.
- **C** – Clear the canvas.
- **Q** – Quit the application.

Ensure proper lighting and keep your index finger clearly visible.

## Deployment

###  Option A – Using PowerShell script

```bash
PowerShell -ExecutionPolicy Bypass -File .\AI_gesture-2--main\build_exe.ps1 -Clean
```

This creates:

```text
AI_gesture-2--main/AI_gesture-2--main/dist/hand_gesture_recognition.exe
```

###  Option B – Manual PyInstaller build

```bash
pyinstaller --noconfirm --onefile --name hand_gesture_recognition \
  --add-data "model/model/hand_gesture_model.h5;model/model" \
  --add-data "model/labels.txt;model" \
  --collect-all mediapipe --collect-all numpy --collect-all cv2 \
  hand_gesture_recognition.py
```

Run the generated executable from:

```bash
./dist/hand_gesture_recognition.exe
```

##  Optimizations

- Smooth strokes through interpolation between index finger points.
- Square padding before resizing to maintain aspect ratio.
- Left-to-right contour sorting for correct word formation.
- User-triggered speech avoids accidental repeats.
- Thicker strokes for better segmentation clarity.

##  Used By

This project can be used by:

- People with speech disabilities or difficulties, enabling them to communicate more easily through air-written gestures and offline text-to-speech.
- Researchers and students exploring gesture recognition, assistive technologies, and human-computer interaction.

## Appendix

- **Model Input**: 28×28 grayscale images.
- **Labels**: Uppercase A–Z and digits 0–9 (subset supported by the trained model).
- **Canvas Segmentation**: Contour-based left-to-right sorting with padding.
- **Repeat Policy**: Speech is triggered manually to prevent unintended outputs.

##  Known Issues

- Camera feed optimization may be needed on some devices.
- Audio output inconsistencies depending on system setup and voices.

##  Troubleshooting

- **PowerShell blocks scripts**: Run the following in the current session:
  ```bash
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

- **Path length errors**: Move the project to a shorter path, e.g., `C:\projects\gesture`.

- **Model not found**: Verify that `model/model/hand_gesture_model.h5` and `model/labels.txt` exist before building.

##  Relevant Files

- `hand_gesture_recognition.py`: Main application code.
- `gesture_utils.py`: Image preprocessing and segmentation.
- `model/`: Trained model and labels.
- `build_exe.ps1`: PowerShell build script.
- `hand_gesture_recognition.spec`: PyInstaller configuration.
- `requirements.txt`: Python dependencies.
- `dist/`: Contains the built executable.
