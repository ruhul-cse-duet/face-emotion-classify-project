import streamlit as st
import time
import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
import os

import torch
import numpy as np
from PIL import Image

from src.custom_resnet import prediction_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_labels = ['Fear', 'Surprise', 'Angry', 'Sad', 'Happy']

emotion_descriptions = {
    'Angry': (
        "😠 Tension across the brow or jaw indicates heightened arousal.",
        "Offer decompression exercises or tag the clip for escalation review."
    ),
    'Fear': (
        "😨 Raised brows and widened eyes usually signal fear or anxiety.",
        "Provide reassurance cues or log the moment for sentiment tracking."
    ),
    'Happy': (
        "😊 Relaxed eyes and lifted cheeks suggest positive engagement.",
        "Celebrate the interaction or store as a positive training sample."
    ),
    'Sad': (
        "😔 Downward gaze or lip corners often align with sadness.",
        "Consider empathy workflows or proactive outreach."
    ),
    'Surprise': (
        "😮 Sudden eye and mouth widening typically indicate surprise.",
        "Verify the cause—surprise can precede both delight and concern."
    )
}

st.set_page_config(page_title="Face Emotion Classification Dashboard", layout="wide")
with open("assets/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0
app_mode = st.session_state['page']
if 'pred_label' not in st.session_state:
    st.session_state['pred_label'] = None
if 'probabilities' not in st.session_state:
    st.session_state['probabilities'] = None



if(app_mode == "home"):
    st.markdown('<h1 class="title"; style="text-align:center; margin: 0.5rem 0;">Face Emotion Classification</h1>', unsafe_allow_html=True)
    st.markdown('<style>[data-testid="stSidebar"]{display:none;}</style>', unsafe_allow_html=True)
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.markdown(
            """
            <div class="hero">
              <div>
                <p class="hero-t">
                    Monitor facial sentiment with a lightweight CNN distilled from ResNet backbones.
                    Upload a webcam frame or still image to detect dominant emotions (happy, sad, angry, fear, surprise) in seconds.
                    Analysts, UX researchers, and mental well-being teams can rapidly triage user sessions with confidence overlays.
                    This demo is for educational insight only—always validate before making sensitive decisions.
                </p>
                <div class="cta"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Emotions we detect", expanded=True):
            st.markdown(
                "- **Happy**: positive affect and engaged presence.\n"
                "- **Sad**: subdued expression with downward gaze.\n"
                "- **Angry**: tightened facial muscles and brows.\n"
                "- **Fear**: widened eyes and tension around mouth.\n"
                "- **Surprise**: rapid widening of eyes and mouth."
            )

        start = st.button("Analyze Emotion", type="primary")
        if start:
            st.session_state['page'] = 'analysis'
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.session_state['probabilities'] = None
            st.rerun()
        st.markdown(
            """
            <div class="card-grid">
                <div class="card"><h3>Edge Ready</h3><p>
                    Custom CNN distilled from ResNet ideas; runs on CPU and scales with CUDA.</p>
                </div>
                <div class="card"><h3>Actionable Insights</h3><p>
                    Confidence scores plus suggested next steps for each detected emotion.</p>
                </div>
            </div>
            """,
             unsafe_allow_html=True,
        )
    with colB:
        sample_image_path = "test_img/images.jpeg"
        if os.path.exists(sample_image_path):
            st.image(sample_image_path, caption="Sample Emotion Image", width='stretch')
        else:
            st.info("Sample image not found. Please upload a face photo to test the application.")

    st.markdown(
        """
             <div class="footer">For demo use only — not a substitute for professional evaluation.</div>
        """, unsafe_allow_html=True,
    )
elif(app_mode=="analysis"):
    nav_cols = st.columns([0.2, 0.6, 0.2])
    with nav_cols[0]:
        if st.button("Home"):
            st.session_state['page'] = 'home'
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.session_state['probabilities'] = None
            st.rerun()
    with nav_cols[1]:
        st.markdown('<div id="ripeness-analysis-anchor"></div>', unsafe_allow_html=True)
    with nav_cols[2]:
        pass

    st.markdown('<div id="ripeness-analysis"><p>Face Emotion Classification</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded is None:
        st.info("Please upload a face photo (JPEG/PNG) to begin the analysis.")
    else:
        image = Image.open(uploaded).convert('RGB')
        display_image = image.resize((400,400))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(display_image, caption="Uploaded Image", width=400)
            c1, c2 = st.columns(2)
            with c1:
                predict_clicked = st.button("Predict", type="primary")
            with c2:
                clear_clicked = st.button("Clear")

        if clear_clicked:
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.session_state['probabilities'] = None
            st.rerun()

        if predict_clicked:
            image_resized = image.resize((384,384))
            img_array = np.array(image_resized).astype(np.float32) / 255.0
            if img_array.ndim == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            def to_device(data, device):
                if isinstance(data, (list,tuple)):
                    return [to_device(x, device) for x in data]
                return data.to(device, non_blocking=True)

            x = img_tensor.unsqueeze(0)
            x = to_device(x, device)

            with st.spinner("Running prediction..."):
                start = time.time()
                pred_idx_tensor, prob_tensor = prediction_img(x)

                if isinstance(pred_idx_tensor, torch.Tensor):
                    pred_idx_tensor = pred_idx_tensor.squeeze()
                if isinstance(pred_idx_tensor, (list, tuple)):
                    pred_idx = pred_idx_tensor[0]
                else:
                    pred_idx = pred_idx_tensor

                pred_idx = int(pred_idx)

                pred_label = class_labels[pred_idx]
                st.session_state['pred_label'] = pred_label

                if isinstance(prob_tensor, torch.Tensor):
                    prob_vector = prob_tensor.squeeze().tolist()
                else:
                    prob_vector = prob_tensor

                if isinstance(prob_vector, float):
                    prob_vector = [prob_vector]

                st.session_state['probabilities'] = {
                    label: float(prob_vector[i]) if i < len(prob_vector) else 0.0
                    for i, label in enumerate(class_labels)
                }

                end = time.time()
                logging.info(f"Prediction Response Time: {end - start:.4f} sec")

        with col2:
            if st.session_state['pred_label']:
                label = st.session_state['pred_label']
                st.image(display_image, caption=f"Prediction: {label}", width=400)
                summary, action = emotion_descriptions.get(label, ("", ""))
                if summary:
                    st.success(summary)
                if action:
                    st.info(action)
                probs = st.session_state.get('probabilities')
                if probs:
                    confidence = probs.get(label, 0.0)
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    st.write("Emotion probabilities")
                    for lbl, prob in probs.items():
                        st.progress(int(min(max(prob, 0.0), 1.0) * 100), text=lbl)
                        st.caption(f"{lbl}: {prob*100:.1f}%")
            else:
                st.image(display_image, caption="Prediction pending", width=400)
                st.info("Click Predict to estimate the dominant emotion.")
    st.markdown(
        """
             <div class="footer">For demo use only — not a substitute for professional evaluation.</div>
        """, unsafe_allow_html=True,
    )

# streamlit run App.py