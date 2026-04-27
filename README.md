# 🩺 X-ray Disease Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?logo=streamlit)]()
[![Status](https://img.shields.io/badge/Status-Active-success)]()

---

## 🌐 Live Demo

🚀 **Try the app here:**
👉 https://x-ray-lungs-disease-detection-us.streamlit.app/

---

## 🚀 Project Overview

This project is an AI-powered system that detects diseases from chest X-ray images using deep learning.
It classifies images into:

* ✅ **Normal**
* ⚠️ **Diseased**

The model is built using **ResNet18 (Transfer Learning)** and deployed with **Streamlit** for real-time predictions.

---

## 🧠 Key Features

* Deep learning-based medical image classification
* Transfer learning using ResNet18
* Handles imbalanced dataset using weighted loss
* Threshold tuning for better disease detection
* Real-time prediction via Streamlit
* Confidence score + risk level visualization

---



## 🧩 Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Model:** ResNet18 (Transfer Learning)
* **Libraries:** NumPy, Pandas, PIL, Matplotlib
* **Deployment:** Streamlit
* **Hardware:** GPU (CUDA)

---

## 📂 Project Structure

```bash
├── app.py                 # Streamlit application
├── requirements.txt      # Dependencies
├── samples/              # Sample images (optional)
├── README.md             # Documentation
```

---

## ⚙️ How It Works

1. Upload an X-ray image
2. Image is resized to **128×128**
3. Passed through **ResNet18 model**
4. Sigmoid converts output → probability
5. Threshold tuning applied
6. Final prediction displayed

---

## 📊 Model Details

* **Architecture:** ResNet18
* **Output:** Binary (1 neuron)
* **Loss Function:** BCEWithLogitsLoss
* **Optimizer:** Adam
* **Image Size:** 128×128

---

## ⚠️ Challenges & Solutions

### 🔴 Imbalanced Dataset

* Majority: Normal
* Minority: Diseased

✅ Solution:

* Used **pos_weight**
* Applied **threshold tuning**

---

### 🔴 Low Probability Outputs

✅ Solution:

* Applied **scaling for UI visualization**
* Used **percentile-based thresholding**

---

## 📈 Evaluation Summary

* **Accuracy:** ~83%
* **Precision (Disease):** ~12%
* **Recall (Disease):** ~13%
* **F1 Score:** ~13%

> ⚠️ Accuracy can be misleading due to dataset imbalance. Recall is prioritized.

---

## 🌐 Deployment

The model is deployed using **Streamlit**:

* Upload X-ray image
* Get real-time prediction
* View confidence score
* See risk level (Low / Medium / High)

---

## 📦 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/xray-disease-detection.git
cd xray-disease-detection
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model file

Place your trained model:

```bash
best_model.pth
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---



---

## 🔮 Future Improvements

* Train on full dataset
* Multi-class classification
* Grad-CAM visualization
* Cloud deployment
* Improve recall for disease class

---

## 📌 Disclaimer

This project is for **educational purposes only** and should not be used for real medical diagnosis.

---

## 👨‍💻 Author

**Utkarsh Sharma**
B.Tech CSE (AI)

---

## ⭐ Support

If you like this project:
👉 Star ⭐ the repo
👉 Share with others
