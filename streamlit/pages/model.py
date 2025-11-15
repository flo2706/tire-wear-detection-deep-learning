import streamlit as st
import matplotlib.pyplot as plt
import json
import tensorflow as tf

# -----------------------------------------------------
#  GLOBAL STYLE: center all titles (H1 → H6)
# -----------------------------------------------------
st.markdown(
    """
    <style>
        /* Center only H1 & H2 */
        h1, h2 {
            text-align: center !important;
        }

        /* Left-align all other subtitles */
        h3, h4, h5, h6 {
            text-align: left !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- BASELINE MODEL: INCEPTIONV3 ----------------------

st.title("Présentation des modèles 🤖")

# Download training history for InceptionV3 baseline
try:
    history_path = tf.keras.utils.get_file(
        "inceptionV3modelHistory.json",
        origin=(
            "https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/"
            "resolve/main/inceptionV3modelHistory.json"
        ),
        cache_subdir="models",
    )
except Exception as e:
    st.error(f"L'historique du modèle baseline n'a pas pu être téléchargé : {e}")
    st.stop()

# Load metrics from JSON history file
with open(history_path, "r") as f:
    metrics = json.load(f)

st.markdown("""
Le modèle initial est un modèle de deep learning utilisant du transfer learning, 
c'est-à-dire que l'architecture repose sur **InceptionV3**, pré-entraîné sur *ImageNet*.
""")

st.markdown("""
Stockage du modèle : https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/tree/main  
Téléchargement : [inceptionV3model.h5 🦊](https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/resolve/main/inceptionV3model.h5)
""")

st.subheader("Schéma du modèle 🪪")

st.markdown(r"""
Le modèle a été entraîné avec l'optimizer `Adam` (lr = $10^{-5}$), 
et une `BinaryCrossentropy` + `BinaryAccuracy`.
""")

st.subheader("Performances du modèle 📈")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["binary_accuracy"], c="r", label="train_accuracy")
    ax.plot(metrics["val_binary_accuracy"], c="b", label="val_accuracy")
    ax.set_title("Accuracy — Entraînement vs Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics["loss"], c="r", label="train_loss")
    ax.plot(metrics["val_loss"], c="b", label="val_loss")
    ax.set_title("Loss — Entraînement vs Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

st.write("")
st.markdown("Matrice de confusion :")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("pages/confusion_matrix/matrix_confusion_inceptionv3.png", use_container_width=True)

# ---------------------- FINAL MODEL: MOBILENETV2 FINE-TUNED ----------------------

st.title("Présentation et informations sur le modèle final 🤖")

# Download training history for final model
try:
    history_path2 = tf.keras.utils.get_file(
        "mobilenetv2_fine_tune_History.json",
        origin=(
            "https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/"
            "resolve/main/mobilenetv2model_finetune_History.json"
        ),
        cache_subdir="models",
    )
except Exception as e:
    st.error(f"L'historique du modèle final n'a pas pu être téléchargé : {e}")
    st.stop()

with open(history_path2, "r") as f:
    metrics2 = json.load(f)

st.markdown("""
Le modèle final utilise **MobileNetV2** pré-entraîné sur *ImageNet*, 
avec un **fine-tuning des 10 dernières couches**.  
Nous avons retenu le **meilleur checkpoint (epoch 23)**, val_acc = 0.8622.
""")

st.markdown("""
Stockage : https://huggingface.co/HyraXuna/JET_model_MobileNetV2/tree/main  

- Dernière epoch : [mobilenetv2model_finetune.h5 🦊](https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/mobilenetv2model_finetune.h5)  
- Meilleur checkpoint (epoch 23) : [model_epoch_23_val_acc_0.86.h5 🐯](https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/model_epoch_23_val_acc_0.86.h5)
""")

st.subheader("Schéma du modèle 🪪")


st.markdown(r"""
Optimizer : `Adam` (lr = $10^{-5}$)  
Loss : `BinaryCrossentropy`  
Early stopping : patience = 5  
Fine-tuning : dernières 10 couches débloquées  
""")

st.subheader("Performances du modèle 📈")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics2["binary_accuracy"], c="r", label="train_accuracy")
    ax.plot(metrics2["val_binary_accuracy"], c="b", label="val_accuracy")
    ax.set_title("Accuracy — Entraînement vs Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics2["loss"], c="r", label="train_loss")
    ax.plot(metrics2["val_loss"], c="b", label="val_loss")
    ax.set_title("Loss — Entraînement vs Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("pages/confusion_matrix/matrix_confusion_mobilinetv2_finetune.png", use_container_width=True)

# ---------------------- YOLOV8 MODEL ----------------------

st.title("Modèle de détection : YOLOv8")

st.markdown("""
YOLOv8 (Ultralytics) est utilisé pour **détecter automatiquement les pneus** 
dans une image et les entourer avec des bounding boxes.  
Le dataset Roboflow fournit images + labels au format YOLOv8.
""")

st.subheader("Détails techniques")

st.markdown("""
| Paramètre           | Valeur |
|--------------------|--------|
| Modèle             | YOLOv8 (Medium) |
| Taille des images  | 800×800 |
| Epochs             | 50 |
| Poids              | [best.pt](https://huggingface.co/flodussart/jet_yolov8m/resolve/main/best.pt) |
| Config             | [data.yaml](https://huggingface.co/datasets/flodussart/tires_project_roboflow/blob/main/data.yaml) |
| Classe(s)          | 1 (pneu) |
""", unsafe_allow_html=True)

st.subheader("Performances 📈")

st.markdown("""
| Metric      | Validation | Test |
|-------------|------------|------|
| Precision   | 93.32%     | 96.85% |
| Recall      | 93.66%     | 91.56% |
| mAP@50      | 97.37%     | 97.30% |
| mAP@50-95   | 61.99%     | 61.65% |
""", unsafe_allow_html=True)

st.markdown("""
YOLOv8 détecte **très efficacement les pneus**, même dans des contextes variés.
""")

st.markdown("---")
st.subheader("Pipeline OpenCV après détection")

st.markdown("""
Après la détection, OpenCV est utilisé pour :
- 📐 Redimensionner les zones détectées  
- 🎨 Convertir BGR → RGB  
- 🧩 Découper chaque pneu en grille 4×4  

Cela permet une **analyse locale de l’usure**, zone par zone.
""")
