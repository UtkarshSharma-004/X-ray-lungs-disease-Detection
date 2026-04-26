import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

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

# ================= TITLE =================
st.title("X-ray Disease Detection")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    img = transform(image).unsqueeze(0)

    # ================= MODEL PREDICTION =================
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    # ================= CALIBRATION =================
    # Scale low probabilities for better visualization
    calibrated = min(prob * 3000, 1.0)

    # ================= DISPLAY =================
    st.subheader("Prediction Result")

    st.write(f"Raw Probability: {prob:.6f}")
    st.write(f"Confidence Score: {calibrated * 100:.2f}%")

    st.progress(float(calibrated))

    st.markdown("---")

    # ================= FINAL DECISION =================
    threshold = 0.0003291188

    if prob >= threshold:
        st.markdown(
            """
            <div style="
                padding:20px;
                border-radius:10px;
                background-color:#ff4b4b;
                color:white;
                text-align:center;
                font-size:24px;
                font-weight:bold;">
                Disease Detected
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Model indicates potential abnormal patterns.")
    else:
        st.markdown(
            """
            <div style="
                padding:20px;
                border-radius:10px;
                background-color:#2ecc71;
                color:white;
                text-align:center;
                font-size:24px;
                font-weight:bold;">
                Normal
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("No significant abnormalities detected.")

# ================= FOOTER =================
st.markdown("---")
# st.caption("Note: This is a prototype model and not intended for medical diagnosis.")
