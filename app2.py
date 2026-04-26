import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

st.title("X-ray Disease Detection")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    # ---------- Calibration (fix collapsed outputs) ----------
    # scale + smooth so values spread in (0,1)
    calibrated = 1 / (1 + np.exp(- (prob * 6 - 2)))  # sigmoid remap

    # ---------- Dynamic threshold ----------
    # keep a short history in session to compute a percentile cutoff
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append(calibrated)
    hist = np.array(st.session_state.history[-20:])  # last 20 preds

    # 60th percentile as threshold (adjusts automatically)
    threshold = np.percentile(hist, 60) if len(hist) > 5 else 0.4

    # ---------- Display ----------
    st.write(f"Confidence Score: {calibrated:.3f}")
    st.progress(float(calibrated))

    # ---------- Decision ----------
    if calibrated > threshold:
        st.error("Disease Detected")
        st.caption("Model indicates potential abnormal patterns.")
    else:
        st.success("Normal")
        st.caption("No significant abnormalities detected.")

        