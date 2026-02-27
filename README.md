# EMNIST Alphanumeric Recognizer

A web application that recognizes handwritten characters (letters and digits) using a CNN model trained on the EMNIST Balanced dataset.

## Tech Stack
- **Model**: TensorFlow/Keras CNN
- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JS + Chart.js

## Project Structure
```
emnist-project/
├── backend/
│   ├── app.py
│   ├── model.h5
│   └── mapping.json
├── frontend/
│   └── index.html
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
conda create -n emnist python=3.10
conda activate emnist
pip install flask flask-cors tensorflow pillow numpy
```

### 2. Run backend
```bash
cd backend
python app.py
```

### 3. Open frontend
Open `frontend/index.html` in browser

## Model Details
- **Dataset**: EMNIST Balanced (47 classes)
- **Architecture**: CNN with BatchNormalization + Dropout
- **Test Accuracy**: 89.56%


## Features
- Interactive drawing canvas
- Real-time character recognition
- Top 3 predictions with confidence bar chart

- Support for digits (0-9) and letters (A-Z, a-z)

https://drive.google.com/file/d/19RoVlQnwu8NQw4Qbsx6hH7gqXFyiobxi/view?usp=sharing
