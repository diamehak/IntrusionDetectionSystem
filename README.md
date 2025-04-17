# 🔐 Intrusion Detection System using Machine Learning

A network-based Intrusion Detection System (IDS) that uses machine learning to classify network traffic and detect potential intrusions in real time or from log datasets.

## 🚀 Features
- 📊 Uses the CICIDS2017 dataset for realistic attack simulation
- 🔍 Trains a Random Forest classifier on preprocessed network data
- 📡 (Optional) Supports real-time packet capture using PyShark
- 🧠 Classifies packets as benign or malicious
- 💾 Saves and loads the trained model (`ids_model.pkl`) for deployment

## 🧰 Tech Stack
- Python
- scikit-learn
- pandas
- pyshark (for live monitoring)
- joblib

## 📂 Directory Structure
```bash
intrusion-detection-system/
├── data/                 # Dataset files
├── model/                # Trained ML model
├── src/                  # Source scripts for training and prediction
├── utils/                # Feature extraction (custom logic)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
