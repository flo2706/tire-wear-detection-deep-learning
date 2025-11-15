import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
import keras

# ---------------------- ML FUNCTIONS ----------------------

def preprocess_zone(cell, size=(224, 224)):
    """Preprocess a sub-image (zone) for MobileNetV2."""
    cell = cv2.resize(cell, size, interpolation=cv2.INTER_AREA)
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    array = img_to_array(cell)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)

def predict_wear(cells, model, seuil=0.6):
    """Predict wear level (worn/good) for each zone using the classifier."""
    labels, confidences = [], []
    for cell in cells:
        input_img = preprocess_zone(cell)
        pred = model.predict(input_img, verbose=0)

        # Binary classifier case (output shape = [1])
        if pred.shape[1] == 1:
            cls = int(pred[0][0] > seuil)
            conf = pred[0][0] if cls == 1 else 1 - pred[0][0]
        else:
            # Multi-class case
            cls = np.argmax(pred)
            conf = pred[0][cls]

        labels.append(cls)
        confidences.append(conf)

    return labels, confidences

def grid_split_img(image, rows=4, cols=4):
    """Split tire image into a grid (rows × cols) with small margins."""
    h, w, _ = image.shape
    cell_h, cell_w = h // rows, w // cols
    margin = 10
    cells = []

    for i in range(rows):
        for j in range(cols):
            y1, x1 = i * cell_h, j * cell_w
            y2, x2 = y1 + cell_h, x1 + cell_w

            # Add margin but ensure bounds stay valid
            y1, y2 = max(y1 - margin, 0), min(y2 + margin, h)
            x1, x2 = max(x1 - margin, 0), min(x2 + margin, w)

            cell = image[y1:y2, x1:x2]
            cells.append(cell)

    return cells

def show_prediction_grid(zones, labels, confidences, class_names):
    """Display the grid of zones with model predictions and confidence values."""
    st.markdown("<div class='centered-grid'>", unsafe_allow_html=True)
    cols = st.columns(4)

    for i, cell in enumerate(zones):
        label = class_names[labels[i]]
        conf = confidences[i]
        color = "🟩" if label == "bon" else "🟥"

        with cols[i % 4]:
            st.image(
                Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)),
                width=120
            )
            st.markdown(
                f"<div style='text-align:center'><b>Zone {i+1}</b><br>"
                f"{color} {label} ({conf:.2f})</div>",
                unsafe_allow_html=True
            )

# ---------------------- MODEL LOADING (CACHED) ----------------------

@st.cache_resource
def load_models():
    """Load and cache both the MobileNetV2 classifier and YOLOv8 detector."""

    # Load wear classifier (MobileNetV2 finetuned)
    model_path = keras.utils.get_file(
        "mobilenetv2_finetune.h5",
        origin=(
            "https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/"
            "resolve/main/model_epoch_23_val_acc_0.86.h5"
        ),
        cache_subdir="models",
    )

    wear_model = keras.saving.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    # Load YOLOv8 detection model
    yolo_path = hf_hub_download(
        repo_id="flodussart/jet_yolov8m",
        filename="best.pt"
    )
    yolo_model = YOLO(yolo_path)

    return wear_model, yolo_model

# ---------------------- STREAMLIT UI ----------------------

st.markdown(
    """
    <style>
    .centered-grid { max-width: 1200px; margin: 0 auto; }
    div[data-testid="column"] {
        padding: 0.5rem !important;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Analyse de l'usure des pneus")

uploaded_image = st.file_uploader(
    "Téléversez une image (voiture/pneu)",
    type=["jpg", "jpeg", "png"]
)

# ---------------------- MAIN LOGIC ----------------------

if uploaded_image is not None:

    # Load the uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Image originale", width=400)

    # Load models only once (cached)
    with st.spinner("Chargement des modèles..."):
        wear_model, yolo_model = load_models()

    st.subheader("Analyse automatique des pneus détectés")

    # Run YOLO detection
    results = yolo_model(image_cv2)[0]

    if len(results.boxes) > 0:
        for idx, box in enumerate(results.boxes):

            st.markdown(f"### 🛞 Pneu détecté {idx+1}")

            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_cv2.shape[1]), min(y2, image_cv2.shape[0])

            # Crop tire region
            zoom = image_cv2[y1:y2, x1:x2]
            zoom_rgb = cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)

            # Global wear prediction
            zoom_resized = cv2.resize(zoom_rgb, (224, 224))
            zoom_array = img_to_array(zoom_resized)
            zoom_array = tf.keras.applications.mobilenet_v2.preprocess_input(zoom_array)
            zoom_array = np.expand_dims(zoom_array, axis=0)
            global_pred = wear_model.predict(zoom_array, verbose=0)[0][0]
            global_conf = global_pred if global_pred > 0.5 else 1 - global_pred

            # Grid zone analysis
            zones = grid_split_img(zoom)
            labels, confidences = predict_wear(zones, wear_model)
            class_names = ["usé", "bon"]

            # Display results
            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.image(Image.fromarray(zoom_rgb), caption="Pneu détecté", width=250)
                if global_pred < 0.5:
                    st.error(f"❌ Ce pneu semble **usé** ({global_conf:.2f})")
                else:
                    st.success(f"✅ Ce pneu semble **bon** ({global_conf:.2f})")

            with right_col:
                st.markdown("#### Prédiction par zones")
                show_prediction_grid(zones, labels, confidences, class_names)

            st.divider()

    else:
        st.warning("⚠️ Aucun pneu détecté dans l'image.")

else:
    st.info("Déposez une image à gauche pour commencer l’analyse.")
