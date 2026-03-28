import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import pandas as pd
from collections import Counter
import torchvision.transforms as transforms

from streamlit_image_coordinates import streamlit_image_coordinates

from utils.inference import (
    load_yolo,
    load_effnet,
    load_mobilenet,
    yolo_pipeline
)
from utils.risk import compute_risk
from utils.gradcam import generate_gradcam
from utils.pdf_report import generate_pdf

st.set_page_config(layout="wide")
st.title("🌊 Microplastic Risk Intelligence System")

# ---------------- Sidebar ----------------
mode = st.sidebar.radio("Mode", ["Single Image", "Batch Processing"])
show_cam = st.sidebar.checkbox("🔥 Show Explainability (Grad-CAM)")
calibrate = st.sidebar.checkbox("📏 Enable Scale Calibration")

classifier_mode = st.sidebar.selectbox(
    "Classifier Mode",
    ["⚡ MobileNet (Fast)", "🧠 EfficientNet (Accurate)"]
)

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    return {
        "yolo": load_yolo("/Users/aniruddhaambekar/Desktop/synapse26/yolov8n.pt"),
        "effnet": load_effnet("/Users/aniruddhaambekar/Desktop/synapse26/models/microplastic_effnet_v2s (3).pth"),
        "mobilenet": load_mobilenet("/Users/aniruddhaambekar/Desktop/synapse26/models/mobilenet (1).pth")
    }

models = load_models()
clf_model = models["mobilenet"] if "MobileNet" in classifier_mode else models["effnet"]

st.sidebar.info(f"Using: {classifier_mode}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Helper ----------------
def find_clicked_particle(x, y, results):
    for res in results:
        x1, y1, x2, y2 = res["box"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return res
    return None


# =========================================================
# 🟢 SINGLE IMAGE MODE
# =========================================================
if mode == "Single Image":

    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file)
        img_np = np.array(image)

        st.image(image, caption="Input Image", use_container_width=True)

        # -------- Calibration --------
        if calibrate:
            st.subheader("📏 Scale Calibration")

            known_length = st.number_input("Enter known length (µm)", 1.0, value=100.0)

            h, w, _ = img_np.shape

            x1 = st.slider("x1", 0, w, 0)
            y1 = st.slider("y1", 0, h, 0)
            x2 = st.slider("x2", 0, w, w//2)
            y2 = st.slider("y2", 0, h, h//2)

            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            scale_factor = known_length / pixel_distance if pixel_distance > 0 else 1.0

            st.success(f"Scale: {scale_factor:.4f} µm/pixel")
        else:
            scale_factor = 1.0

        # -------- Inference --------
        start = time.time()
        results = yolo_pipeline(models["yolo"], clf_model, img_np, scale_factor)
        end = time.time()

        overlay = img_np.copy()
        total_risk = 0

        for res in results:
            x1, y1, x2, y2 = res["box"]
            label = res["label"]
            conf = res["confidence"]
            size = res["size"]
            contour = res.get("contour", None)
            feret_pts = res.get("feret_pts", None)

            crop = img_np[y1:y2, x1:x2]

            risk = compute_risk(label, size)
            total_risk += risk

            text = f"{label} | {size:.1f}µm"

            # -------- Grad-CAM --------
            if show_cam and crop.size != 0:
                img_tensor = transform(crop).unsqueeze(0)
                cam = generate_gradcam(clf_model, img_tensor)
                cam = cam / (cam.max() + 1e-8)

                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))

                overlay[y1:y2, x1:x2] = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)

            # -------- Draw Contour --------
            if contour is not None:
                contour_shifted = contour + np.array([[x1, y1]])
                cv2.drawContours(overlay, [contour_shifted], -1, (0,255,255), 2)

            # -------- Draw Feret Line --------
            if feret_pts is not None:
                p1, p2 = feret_pts
                cv2.line(
                    overlay,
                    (x1 + p1[0], y1 + p1[1]),
                    (x1 + p2[0], y1 + p2[1]),
                    (255,255,0),
                    2
                )

            # -------- Draw Box --------
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(overlay, text,
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0), 2)

        # Draw calibration line
        if calibrate:
            cv2.line(overlay, (x1, y1), (x2, y2), (255,0,0), 2)

        st.subheader("🧪 Interactive Detection")
        coords = streamlit_image_coordinates(overlay)

        # -------- CLICK INTERACTION --------
        if coords:
            selected = find_clicked_particle(coords["x"], coords["y"], results)

            if selected:
                st.subheader("🔍 Particle Inspector")

                x1, y1, x2, y2 = selected["box"]
                crop = img_np[y1:y2, x1:x2]

                st.image(crop, caption="Zoomed Particle")

                st.write(f"**Type:** {selected['label']}")
                st.write(f"**Confidence:** {selected['confidence']:.2f}")
                st.write(f"**Size (Feret):** {selected['size']:.2f} µm")

                risk = compute_risk(selected["label"], selected["size"])
                st.write(f"**Risk Score:** {risk}/100")

                if show_cam:
                    img_tensor = transform(crop).unsqueeze(0)
                    cam = generate_gradcam(clf_model, img_tensor)
                    cam = cam / (cam.max() + 1e-8)

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))

                    cam_overlay = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)

                    st.image(cam_overlay, caption="🔥 Explainability")

        st.write(f"⏱️ Inference Time: {end-start:.2f}s")

        if results:
            avg_risk = int(total_risk / len(results))

            st.subheader("🚨 Overall Ecological Threat")
            st.progress(avg_risk)
            st.write(f"**Risk Score:** {avg_risk}/100")

        else:
            st.warning("No particles detected.")