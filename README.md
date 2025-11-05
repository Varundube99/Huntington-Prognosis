# 🧠 Huntington’s Disease Prognosis

### ML-based system for predicting Huntington’s disease progression stages using machine learning models

---

## 📘 Overview

Huntington’s Disease (HD) is a rare neurodegenerative disorder that leads to the progressive breakdown of nerve cells in the brain.  
This project builds a **machine learning–based tool** to assist in predicting the **disease stage** based on patient data, clinical measurements, and derived features.

The application combines various ML models and ensemble methods to make stage-wise predictions and presents an **interactive Streamlit interface** for clinicians and researchers.

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Varundube99/Huntington-Prognosis.git
cd Huntington-Prognosis
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run STREAMLIT/app.py
```

> ⚠️ Note: Model files are not shared publicly for data protection and research integrity.

---

## 🧩 Project Structure

```
Huntington-Prognosis/
│
├── Notebooks/                 # Training notebooks for various ML models
│   ├── DT_Training.ipynb
│   ├── LR_Training.ipynb
│   ├── RF_Training.ipynb
│   ├── MLP_Training.ipynb
│   ├── SVM_Training.ipynb
│   ├── XGB_Training.ipynb
│   └── Stacked(LR+MLP+XGB).ipynb
│
├── STREAMLIT/                 # Streamlit app files
│   ├── app.py
│   ├── HD1.png / HD2.png / brain.png
│
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files and sensitive data rules
```

---

## 🧬 Features

- 📊 Exploratory Data Analysis (EDA) and feature engineering  
- 🤖 Multi-model training and comparison  
- 🧩 Ensemble prediction (Stacked LR + MLP + XGB)  
- 🎨 Streamlit-based interactive interface  

---

## 🌐 Deployment

The project is deployed on **Streamlit Cloud** for public access:  
👉 [https://huntington-prognosis.streamlit.app](https://huntington-prognosis.streamlit.app)

---

## 🔐 Data & Preprocessing Access

The **dataset** and **preprocessing scripts** used for training are **not publicly shared** to protect data confidentiality and maintain research ethics.

If you are a **researcher, collaborator, or reviewer** who wishes to reproduce or validate this work, please contact the authors directly:

📧 **Contact:** [varundube99@gmail.com](mailto:varundube99@gmail.com)

Access may be granted for **academic or non-commercial** use upon request.

---

## ⚖️ License

This work is licensed under the  
**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).**

> You may **view** and **cite** this repository but **not modify, redistribute, or use it commercially.**

---

## 👨‍🔬 Authors & Contributors

This project was developed as part of an academic research initiative on Huntington’s Disease prognosis using machine learning.

| Name | Contact |
|------|----------|
| **Varun Dubey** | [varundube99@gmail.com](mailto:varundube99@gmail.com) |
| **Harshit Yadav** | [harshityadav0126@gmail.com](mailto:harshityadav0126@gmail.com) |
| **Vishal Gangwar** | [vishalgangwar953@gmail.com](mailto:vishalgangwar953@gmail.com) |

---

⭐ *If you found this project helpful, consider giving it a star on GitHub!*
