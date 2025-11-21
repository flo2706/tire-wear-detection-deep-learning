import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
import keras


# ==============================================================
#                     ML UTILITY FUNCTIONS
# ==============================================================

def preprocess_zone(cell, size=(224, 224)):
    """
    Preprocess a cropped tire zone for MobileNetV2.
    """
    cell = cv2.resize(cell, size, interpolation=cv2.INTER_AREA)
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    array = img_to_array(cell)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict_wear(cells, model, seuil=0.6):
    """
    Predict wear class and confidence for each cell/zone.
    """
    labels = []
    confidences = []

    for cell in cells:
        input_img = preprocess_zone(cell)
        pred = model.predict(input_img, verbose=0)

        # Binary classifier case
        if pred.shape[1] == 1:
            cls = int(pred[0][0] > seuil)
            conf = pred[0][0] if cls == 1 else 1 - pred[0][0]
        else:
            # Multiclass fallback (not used here, but kept for robustness)
            cls = int(np.argmax(pred))
            conf = float(pred[0][cls])

        labels.append(cls)
        confidences.append(conf)

    return labels, confidences


def grid_split_img(image, rows=4, cols=4):
    """
    Split detected tire image into a grid with a small margin.
    """
    height, width, _ = image.shape
    cell_h, cell_w = height // rows, width // cols
    margin = 10
    cells = []

    for i in range(rows):
        for j in range(cols):
            y1, x1 = i * cell_h, j * cell_w
            y2, x2 = y1 + cell_h, x1 + cell_w

            # Add margin while staying inside image bounds
            y1, y2 = max(y1 - margin, 0), min(y2 + margin, height)
            x1, x2 = max(x1 - margin, 0), min(x2 + margin, width)

            cell = image[y1:y2, x1:x2]
            cells.append(cell)

    return cells


def show_prediction_grid(zones, labels, confidences, class_names):
    """
    Display the 4×4 prediction grid for each detected tire.
    """
    st.markdown("<div class='centered-grid'>", unsafe_allow_html=True)
    cols = st.columns(4)

    for i, cell in enumerate(zones):
        label = class_names[labels[i]]
        conf = confidences[i]
        color = "🟩" if label == "bon" else "🟥"

        with cols[i % 4]:
            st.image(
                Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)),
                width=110,
            )
            st.markdown(
                (
                    f"<div style='text-align:center'><b>Zone {i + 1}</b><br>"
                    f"{color} {label} ({conf:.2f})</div>"
                ),
                unsafe_allow_html=True,
            )


# ==============================================================
#                 MODEL LOADING (CACHED ONCE)
# ==============================================================

@st.cache_resource
def load_models():
    """
    Load MobileNetV2 wear classifier and YOLOv8 detector once.
    """
    # Load classifier (MobileNetV2 fine-tuned)
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
        safe_mode=False,
    )

    # Load YOLOv8 detector
    yolo_path = hf_hub_download(
        repo_id="flodussart/jet_yolov8m",
        filename="best.pt",
    )
    yolo_model = YOLO(yolo_path)

    return wear_model, yolo_model


# ==============================================================
#                          MAIN UI
# ==============================================================

st.markdown(
    """
    <div style="text-align: center; font-size: 3rem; color: gray;">
        Jedha Evaluation Tyres
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='text-align: center; font-size: 2rem; color: gray;'>
        A Convolutional Neural Network Project
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

# Centered logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo/Logo_JET.png", use_container_width=True)

st.write("")

# ==============================================================
#                     ANALYSIS INTRO SECTION
# ==============================================================

st.markdown(
    """
    <div style="text-align: center; margin-top: 1rem; margin-bottom: 1rem;">
        <h2>Analyse de l’usure des pneus</h2>
        <p>
            Ce module permet d’<b>évaluer automatiquement l’état des pneus</b>
            à partir d’une photo.
        </p>
        <p>
            Vous pouvez téléverser une image <b>d’un pneu seul</b> ou
            <b>d’une voiture avec les pneus visibles</b> pour lancer l’analyse 👇
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_image = st.file_uploader(
    "📷 Importer une image (voiture ou pneu — JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

st.markdown("---")

# ==============================================================
#                    MAIN PROCESSING LOGIC
# ==============================================================

if uploaded_image is not None:
    # Convert uploaded image to RGB (PIL) and BGR (OpenCV)
    image = Image.open(uploaded_image).convert("RGB")
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Image originale", width=450)

    # Load models (cached after first call)
    with st.spinner(
        "Chargement des modèles de détection et de classification..."
    ):
        wear_model, yolo_model = load_models()

    st.subheader("Résultats de la détection et de l’analyse")

    # YOLO detection
    results = yolo_model(image_cv2)[0]

    if len(results.boxes) > 0:
        st.info(f"✅ {len(results.boxes)} pneu(x) détecté(s) dans l’image.")

        # Limit the number of displayed tires to keep the page readable
        max_tires = 3

        for idx, box in enumerate(results.boxes[:max_tires]):
            st.markdown(f"### 🛞 Pneu détecté n°{idx + 1}")

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_cv2.shape[1]), min(y2, image_cv2.shape[0])

            # Crop detected tire
            zoom = image_cv2[y1:y2, x1:x2]
            zoom_rgb = cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)

            # Global classification on the whole tire
            zoom_resized = cv2.resize(zoom_rgb, (224, 224))
            zoom_array = img_to_array(zoom_resized)
            zoom_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                zoom_array
            )
            zoom_array = np.expand_dims(zoom_array, axis=0)
            global_pred = wear_model.predict(zoom_array, verbose=0)[0][0]
            global_conf = (
                global_pred if global_pred > 0.5 else 1 - global_pred
            )

            # Grid-based analysis (local zones)
            zones = grid_split_img(zoom)
            labels, confidences = predict_wear(zones, wear_model)
            class_names = ["usé", "bon"]

            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.image(
                    Image.fromarray(zoom_rgb),
                    caption="Pneu détecté (recadré)",
                    width=260,
                )

                if global_pred < 0.5:
                    st.error(
                        "❌ Ce pneu semble **usé** "
                        f"(confiance : {global_conf:.2f})"
                    )
                else:
                    st.success(
                        "✅ Ce pneu semble **en bon état** "
                        f"(confiance : {global_conf:.2f})"
                    )

            with right_col:
                st.markdown("#### Analyse détaillée par zones")
                show_prediction_grid(zones, labels, confidences, class_names)

            st.divider()

        if len(results.boxes) > max_tires:
            st.caption(
                "ℹ️ Seuls les "
                f"{max_tires} premiers pneus détectés sont affichés "
                "pour garder une lecture confortable."
            )

    else:
        st.warning(
            "⚠️ Aucun pneu n’a été détecté dans l’image. "
            "Essayez avec un angle plus proche du pneu."
        )

