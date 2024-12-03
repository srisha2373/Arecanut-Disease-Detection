# Arecanut Disease Detection

A comprehensive project for detecting arecanut diseases using machine learning (ML) and deep learning (DL) models, complemented by an interactive Streamlit application for real-time predictions.

## Features
- **Machine Learning**: SVM-based model for basic predictions.
- **Deep Learning**: Advanced model for enhanced accuracy.
- **Streamlit App**: User-friendly interface for predictions and insights.

## Repository Structure
```markdown
arecanut-disease-detection/
├── models/             # Trained ML and DL models
├── data/               # Sample datasets
├── streamlit_app/      # Streamlit application
├── scripts/            # Training and prediction scripts
├── README.md           # Project overview
└── LICENSE             # License file
```

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/yourusername/arecanut-disease-detection.git
cd arecanut-disease-detection
```

### Install Dependencies
Navigate to the `streamlit_app` folder and install the required packages:
```bash
pip install -r streamlit_app/requirements.txt
```

### Run the Streamlit App
```bash
streamlit run streamlit_app/app.py
```
The app will be available at `http://localhost:8501`.

## Usage

### Train Models
- Train ML model: `python scripts/train_ml.py`
- Train DL model: `python scripts/train_dl.py`

### Make Predictions
- ML: `python scripts/predict_ml.py`
- DL: `python scripts/predict_dl.py`


## License
This project is licensed under the MIT License.

