# 🩺 X-ray Disease Detection using Deep Learning 
you can try it here-> https://x-ray-lungs-disease-detection-us.streamlit.app/

An end-to-end deep learning project that detects diseases from chest X-ray images using a pre-trained ResNet18 model and deploys predictions through a Streamlit web application.

---

## 🚀 Project Overview

This project focuses on building an AI-powered system capable of classifying chest X-ray images into:

* ✅ **Normal**
* ⚠️ **Diseased**

The system uses **transfer learning** with ResNet18 and handles challenges such as **imbalanced datasets** and **low probability outputs** through techniques like weighted loss and threshold tuning.

---

## 🧠 Key Features

* Deep learning-based medical image classification
* Transfer learning using ResNet18
* Handles imbalanced dataset using weighted loss
* Threshold tuning for improved disease detection
* Real-time prediction using Streamlit
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

```
├── app.py                 # Streamlit application
├── requirements.txt      # Dependencies
├── best_model.pth        # Trained model (not included in repo)
├── samples/              # Sample X-ray images (optional)
├── README.md             # Project documentation
```

---

## ⚙️ How It Works

1. Upload an X-ray image
2. Image is preprocessed (resized to 128×128)
3. Passed through ResNet18 model
4. Sigmoid converts output to probability
5. Threshold tuning is applied
6. Final prediction displayed with confidence

---

## 📊 Model Details

* **Architecture:** ResNet18
* **Output:** Single neuron (binary classification)
* **Loss Function:** BCEWithLogitsLoss
* **Optimizer:** Adam
* **Image Size:** 128×128

---

## ⚠️ Challenges & Solutions

### Problem: Imbalanced Dataset

* Majority class: Normal
* Minority class: Diseased

### Solution:

* Used **pos_weight** in loss function
* Applied **threshold tuning** instead of default 0.5

---

### Problem: Very Low Output Probabilities

* Model outputs close to 0

### Solution:

* Applied **scaling for visualization**
* Used **percentile-based thresholding**

---

## 📈 Evaluation Summary

* **Accuracy:** ~83%
* **Precision (Disease):** ~12%
* **Recall (Disease):** ~13%
* **F1 Score:** ~13%

> Note: Accuracy is high due to dataset imbalance. Recall is prioritized for medical diagnosis.

---

## 🌐 Deployment

The model is deployed using **Streamlit**, allowing users to:

* Upload X-ray images
* View predictions instantly
* See confidence score and risk level

---

## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/xray-disease-detection.git
cd xray-disease-detection
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model file

Place your trained model file:

```
best_model.pth
```

---

### 5. Run the app

```bash
streamlit run app.py
```

---

## ⚠️ Important Note

* The trained model file is not included due to size limitations (>25MB)
* You can load it via:

  * Google Drive
  * gdown
  * or local file

---

## 🔮 Future Improvements

* Train on full dataset for better accuracy
* Multi-class disease classification
* Add Grad-CAM visualization
* Improve model generalization
* Deploy on cloud

---

## 📌 Disclaimer

This project is for **educational purposes only** and is not intended for real medical diagnosis.

---

## 👨‍💻 Author

**Utkarsh Sharma**
B.Tech CSE (AI)

---

## ⭐ If you like this project

Give it a star on GitHub!
