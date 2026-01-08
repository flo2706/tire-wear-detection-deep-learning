import json
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf


#  Global styles : titles + cards
st.markdown(
    """
    <style>
        /* Center only H1 (main title "Modèles") */
        h1 {
            text-align: center !important;
        }

        /* Other headings left-aligned for readability */
        h2, h3, h4, h5, h6 {
            text-align: left !important;
        }

        /* Cards for models */
        .model-card {
            background-color: #f9f9f9;
            padding: 16px 20px;
            border-radius: 12px;
            border: 1px solid #e0e0e0;
            margin-bottom: 12px;
            min-height: 260px;     /* Harmonize height */
            display: flex;
            flex-direction: column;
        }

        .model-card h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.6rem;
        }

        .model-card p {
            margin-bottom: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load training history from Hugging Face

def load_history(url: str, local_name: str) -> dict:
    """Download and load a JSON history file from Hugging Face."""
    history_path = tf.keras.utils.get_file(
        local_name,
        origin=url,
        cache_subdir="models",
    )
    with open(history_path, "r") as f:
        return json.load(f)


# InceptionV3 (baseline model)
try:
    metrics = load_history(
        url=(
            "https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/"
            "resolve/main/inceptionV3modelHistory.json"
        ),
        local_name="inceptionV3modelHistory.json",
    )
except Exception as e:
    st.error(
        f"L'historique du modèle baseline n'a pas pu être téléchargé : {e}"
    )
    st.stop()

# MobileNetV2 (final model)
try:
    metrics2 = load_history(
        url=(
            "https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/"
            "resolve/main/mobilenetv2model_finetune_History.json"
        ),
        local_name="mobilenetv2_fine_tune_History.json",
    )
except Exception as e:
    st.error(
        f"L'historique du modèle final n'a pas pu être téléchargé : {e}"
    )
    st.stop()


# Main title
st.markdown(
    "<h1 style='color: gray; margin-bottom: 0;'>Modèles</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Section 1 : Classification models
st.markdown("## Modèles de classification")
st.markdown("")

colA, colB = st.columns(2)

# Left column : InceptionV3 (baseline) 
with colA:
    # Card description
    st.markdown(
        """
        <div class="model-card">
          <h3>Baseline – InceptionV3</h3>

          <p>
            Modèle de <b>transfer learning</b> basé sur <b>InceptionV3</b>
            pré-entraîné sur <i>ImageNet</i>, puis ré-entraîné sur le
            dataset de pneus (bon / défectueux).
          </p>

          <p><b>🔗 Ressources :</b></p>
          <ul>
            <li>
              <a href="https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/tree/main"
                 target="_blank">
                Dépôt Hugging Face
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/resolve/main/inceptionV3model.h5"
                 target="_blank">
                Modèle InceptionV3 (.h5) 🦊
              </a>
            </li>
          </ul>

          <p><b>Entraînement :</b></p>
          <ul>
            <li>Optimiseur : <code>Adam</code> (lr = 10<sup>-5</sup>)</li>
            <li>Loss : <code>BinaryCrossentropy</code></li>
            <li>Métrique principale : <code>BinaryAccuracy</code></li>
          </ul>
          </br></br></br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    # Learning curves
    st.markdown("**Courbes d'apprentissage**")

    courbeA1, courbeA2 = st.columns(2)

    # Accuracy plot
    with courbeA1:
        fig, ax = plt.subplots(figsize=(1.8, 1.4))
        ax.plot(metrics["binary_accuracy"], c="r", label="train_accuracy")
        ax.plot(metrics["val_binary_accuracy"], c="b", label="val_accuracy")
        ax.set_title("Accuracy", fontsize=8)
        ax.set_xlabel("Epochs", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        st.pyplot(fig, use_container_width=True)

    # Loss plot
    with courbeA2:
        fig, ax = plt.subplots(figsize=(1.8, 1.4))
        ax.plot(metrics["loss"], c="r", label="train_loss")
        ax.plot(metrics["val_loss"], c="b", label="val_loss")
        ax.set_title("Loss", fontsize=8)
        ax.set_xlabel("Epochs", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        st.pyplot(fig, use_container_width=True)

    st.markdown("")
    # Confusion matrix 
    st.markdown("**Matrice de confusion**")
    cm1_col1, cm1_col2, cm1_col3 = st.columns([1, 2, 1])
    with cm1_col2:
        st.image(
            "pages/confusion_matrix/matrix_confusion_inceptionv3.png",
            width=420,
        )

# Right column : MobileNetV2 (final) 
with colB:
    # Card description 
    st.markdown(
        """
        <div class="model-card">
          <h3>Modèle final – MobileNetV2</h3>

          <p>
            Modèle optimisé basé sur <b>MobileNetV2</b> pré-entraîné sur
            <i>ImageNet</i>, avec <b>fine-tuning des 10 dernières couches</b>.
            Le modèle retenu correspond au <b>meilleur checkpoint (epoch 23)</b> :
            val_acc ≈ 0.86.
          </p>

          <p><b>🔗 Ressources :</b></p>
          <ul>
            <li>
              <a href="https://huggingface.co/HyraXuna/JET_model_MobileNetV2/tree/main"
                 target="_blank">
                Dépôt Hugging Face
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/mobilenetv2model_finetune.h5"
                 target="_blank">
                Dernière epoch (.h5) 🦊
              </a>
            </li>
            <li>
              <a href="https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/model_epoch_23_val_acc_0.86.h5"
                 target="_blank">
                Checkpoint epoch 23 (.h5) 🐯
              </a>
            </li>
          </ul>

          <p><b>Configuration :</b></p>
          <ul>
            <li>Optimiseur : <code>Adam</code> (lr = 10<sup>-5</sup>)</li>
            <li>Loss : <code>BinaryCrossentropy</code></li>
            <li>Early stopping : patience = 5</li>
            <li>Fine-tuning : 10 dernières couches dégelées</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    # Learning curves 
    st.markdown("**Courbes d'apprentissage**")

    courbeB1, courbeB2 = st.columns(2)

    # Accuracy plot
    with courbeB1:
        fig, ax = plt.subplots(figsize=(1.8, 1.4))
        ax.plot(metrics2["binary_accuracy"], c="r", label="train_accuracy")
        ax.plot(metrics2["val_binary_accuracy"], c="b", label="val_accuracy")
        ax.set_title("Accuracy", fontsize=8)
        ax.set_xlabel("Epochs", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        st.pyplot(fig, use_container_width=True)

    # Loss plot
    with courbeB2:
        fig, ax = plt.subplots(figsize=(1.8, 1.4))
        ax.plot(metrics2["loss"], c="r", label="train_loss")
        ax.plot(metrics2["val_loss"], c="b", label="val_loss")
        ax.set_title("Loss", fontsize=8)
        ax.set_xlabel("Epochs", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        st.pyplot(fig, use_container_width=True)
        
    st.markdown("")
    # Confusion matrix 
    st.markdown("**Matrice de confusion**")
    cm2_col1, cm2_col2, cm2_col3 = st.columns([1, 2, 1])
    with cm2_col2:
        st.image(
            "pages/confusion_matrix/matrix_confusion_mobilinetv2_finetune.png",
            width=420,
        )

# Section 2 : Detection model
st.markdown("---")
st.markdown("## Modèle de détection : YOLOv8")

st.markdown(
    """
YOLOv8 (Ultralytics) est utilisé pour **détecter automatiquement les pneus** 
sur l’image et dessiner des *bounding boxes* autour d’eux.  
Le dataset Roboflow fournit les images et annotations au format YOLOv8.
"""
)

st.markdown("### Détails techniques")

st.markdown(
    """
| Paramètre           | Valeur |
|---------------------|--------|
| Modèle              | YOLOv8 (Medium) |
| Taille des images   | 800×800 |
| Epochs              | 50 |
| Poids               | [best.pt](https://huggingface.co/flodussart/jet_yolov8m/resolve/main/best.pt) |
| Config              | [data.yaml](https://huggingface.co/datasets/flodussart/tires_project_roboflow/blob/main/data.yaml) |
| Classe(s)           | 1 (pneu) |
""",
    unsafe_allow_html=True,
)

st.markdown("### Performances 📈")

st.markdown(
    """
| Metric      | Validation | Test |
|-------------|------------|------|
| Precision   | 93.32%     | 96.85% |
| Recall      | 93.66%     | 91.56% |
| mAP@50      | 97.37%     | 97.30% |
| mAP@50-95   | 61.99%     | 61.65% |
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
YOLOv8 détecte **très efficacement les pneus**, y compris dans des scènes complexes
(voitures complètes, angles variés, éclairages différents).
"""
)

st.markdown("---")
st.markdown("### Pipeline OpenCV après détection")

st.markdown(
    """
Après la détection, un pipeline **OpenCV** est appliqué à chaque pneu détecté :

- 📐 Recadrage et redimensionnement de la zone du pneu  
- 🎨 Conversion BGR → RGB  
- 🧩 Découpage du pneu en grille 4×4  

Ce découpage permet une **analyse locale de l’usure**, zone par zone,
avec le classifieur MobileNetV2.
"""
)
