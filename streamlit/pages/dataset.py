import random
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import list_repo_files
from PIL import Image, ImageDraw
from skimage.color import rgb2lab



# Header

st.markdown(
    "<h1 style='text-align: center; color: gray;'>Datasets</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# HTML block for the top description + Bootstrap accordions
components.html(
    """
    <link rel="stylesheet"
          href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js">
    </script>

    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }
      .dataset-card {
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        border: none;
      }
      .dataset-card .card-header {
        background: #f8f9fa;
        border-bottom: 1px solid #eee;
      }
      .dataset-card .btn-link {
        font-size: 1.05rem;
        color: #333;
        text-decoration: none;
        font-weight: 600;
      }
      .dataset-card .btn-link:hover {
        color: #000;
      }
      pre {
        background-color: #f8f9fa;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-bottom: 0;
      }
      ul { padding-left: 1.1rem; }
      .page-title {
        text-align: center;
        margin-bottom: 25px;
      }
      .page-title p {
        margin: 0;
        color: #666;
        font-size: 0.95rem;
      }
    </style>

    <div class="container mt-3 mb-3">
      <div class="page-title">
        <p>
          Deux jeux de données sont utilisés : l’un pour la <strong>classification</strong> de l’état des pneus,
          l’autre pour la <strong>détection</strong> au format YOLOv8.
          </br></br>
        </p>
      </div>

      <div class="row">
        <!-- Classification Dataset -->
        <div class="col-md-6">
          <div id="accordion1">
            <div class="card dataset-card">
              <div class="card-header" id="head1">
                <h5 class="mb-0">
                  <button class="btn btn-link" data-toggle="collapse"
                          data-target="#collapse1" aria-expanded="true">
                    🗂️ Dataset 1&nbsp;: Classification (Inception_v3 / Kaggle)
                  </button>
                </h5>
              </div>
              <div id="collapse1" class="collapse show" data-parent="#accordion1">
                <div class="card-body">
                  <ul>
                    <li><strong>Source Kaggle&nbsp;:</strong>
                      <a href="https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data"
                         target="_blank">Voir sur Kaggle</a>
                    </li>
                    <li><strong>Version Hugging Face&nbsp;:</strong>
                      <a href="https://huggingface.co/datasets/flodussart/tires_project_roboflow"
                         target="_blank">Voir le dataset</a>
                    </li>
                    <li><strong>Auteur&nbsp;:</strong> Chirag CHAUHAN</li>
                    <li><strong>Licence&nbsp;:</strong>
                      <a href="https://creativecommons.org/licenses/by/4.0/"
                         target="_blank">CC BY 4.0</a>
                    </li>
                  </ul>
                  <p>
                    Ce dataset contient des images de pneus réparties en deux classes&nbsp;:
                    <strong>good</strong> (bon état) et <strong>defective</strong> (défectueux).
                    Il est utilisé pour entraîner le modèle de <strong>classification de l’usure</strong>.
                  </p>

                  <pre>
images/
├── defective/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── good/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Detection Dataset -->
        <div class="col-md-6">
          <div id="accordion2">
            <div class="card dataset-card">
              <div class="card-header" id="head2">
                <h5 class="mb-0">
                  <button class="btn btn-link" data-toggle="collapse"
                          data-target="#collapse2" aria-expanded="true">
                    📦 Dataset 2&nbsp;: Détection (YOLOv8 / Roboflow)
                  </button>
                </h5>
              </div>
              <div id="collapse2" class="collapse show" data-parent="#accordion2">
                <div class="card-body">
                  <ul>
                    <li><strong>Source Roboflow&nbsp;:</strong>
                      <a href="https://universe.roboflow.com/iotml/tire-dataset/dataset/2"
                         target="_blank">Voir sur Roboflow</a>
                    </li>
                    <li><strong>Version Hugging Face&nbsp;:</strong>
                      <a href="https://huggingface.co/datasets/flodussart/tires_project"
                         target="_blank">Voir le dataset</a>
                    </li>
                    <li><strong>Titre&nbsp;:</strong> Tire Dataset – Computer Vision Project</li>
                  </ul>
                  <p>
                    Ce dataset est formaté pour <strong>YOLOv8</strong> avec annotations de détection d’objets.
                    Il contient des images réparties en ensemble d’<em>entraînement</em>, de <em>validation</em>
                    et de <em>test</em>.
                  </p>

                  <pre>
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """,
    height=650,
    scrolling=True,
)


# Global styles

st.markdown(
    """
    <style>
        h1, h2, h3, h4, h5, h6 { text-align: center; }
        .markdown-text-container { font-size: 0.85rem !important; }
        .element-container img + div {
            font-size: 0.75rem !important;
            text-align: center;
            color: #555;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Dataset configuration


CLASSIF_REPO_ID = "flodussart/tires_project"
CLASSIF_BASE_URL = (
    f"https://huggingface.co/datasets/{CLASSIF_REPO_ID}/resolve/main/"
)

DETECT_REPO_ID = "flodussart/tires_project_roboflow"
DETECT_BASE_URL = (
    f"https://huggingface.co/datasets/{DETECT_REPO_ID}/resolve/main/"
)


# Helpers to load datasets

@st.cache_data
def get_classification_image_df() -> pd.DataFrame:
    """
    List classification images (good / defective) from the HF dataset.
    """
    files = list_repo_files(CLASSIF_REPO_ID, repo_type="dataset")

    image_paths = [
        file
        for file in files
        if file.endswith((".jpg", ".jpeg", ".png"))
        and ("good/" in file or "defective/" in file)
    ]

    data = []
    for path in image_paths:
        label = "good" if "good/" in path else "defective"
        url = CLASSIF_BASE_URL + path
        data.append({"url": url, "label": label})

    return pd.DataFrame(data)


def compute_color_features(
    df: pd.DataFrame,
    sample_per_class: int = 100,
) -> pd.DataFrame:
    """
    Compute simple color features (RGB + Lab) from a sample of images per class.
    """
    data = []

    for label in ["good", "defective"]:
        class_df = df[df["label"] == label]
        if len(class_df) == 0:
            continue

        n_samples = min(sample_per_class, len(class_df))
        subset = class_df.sample(n=n_samples, random_state=42)

        for _, row in subset.iterrows():
            try:
                response = requests.get(row["url"])
                image = (
                    Image.open(BytesIO(response.content))
                    .convert("RGB")
                    .resize((64, 64))
                )
                arr = np.array(image)
                lab = rgb2lab(arr / 255.0)

                data.append(
                    {
                        "url": row["url"],
                        "label": label,
                        "R": arr[..., 0].mean(),
                        "G": arr[..., 1].mean(),
                        "B": arr[..., 2].mean(),
                        "L": lab[..., 0].mean(),
                        "a": lab[..., 1].mean(),
                        "b": lab[..., 2].mean(),
                    }
                )
            except Exception:
                # Skip problematic images but keep the rest
                continue

    return pd.DataFrame(data)


@st.cache_data
def get_detection_image_paths() -> list[str]:
    """
    List YOLO detection images (train/images) from the HF dataset.
    """
    files = list_repo_files(DETECT_REPO_ID, repo_type="dataset")

    return [
        file
        for file in files
        if file.startswith("train/images/")
        and file.endswith((".jpg", ".jpeg", ".png"))
    ]


# Load data for preview

df_classif = get_classification_image_df()
detect_image_paths = get_detection_image_paths()


# Side-by-side preview of both datasets

st.markdown("---")
st.subheader("Aperçu visuel des deux jeux de données")

df = df_classif
image_paths = detect_image_paths

col_left, col_right = st.columns(2)

# Left : Classification

with col_left:
    st.markdown("**Dataset 1 – Classification**")

    if len(df) > 0:
        # Up to 6 images → 2 rows × 3 columns
        sample_classif = df.sample(min(6, len(df)), random_state=42)
        img_cols = st.columns(3)

        for i, row in enumerate(sample_classif.itertuples()):
            with img_cols[i % 3]:
                st.image(
                    row.url,
                    caption=row.label,
                    width=140,  # small thumbnail
                )
    else:
        st.info("Aucune image trouvée pour le dataset de classification.")


# Right : Detection

with col_right:
    st.markdown("**Dataset 2 – Détection (YOLOv8)**")

    if len(image_paths) > 0:
        sample_detect = random.sample(image_paths, min(6, len(image_paths)))
        img_cols = st.columns(3)

        for i, path in enumerate(sample_detect):
            image_url = f"{DETECT_BASE_URL}{path}"
            label_path = (
                path.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
            )
            label_url = f"{DETECT_BASE_URL}{label_path}"

            try:
                img_response = requests.get(image_url)
                img = Image.open(BytesIO(img_response.content)).convert("RGB")
                width, height = img.size

                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)

                label_response = requests.get(label_url)
                lines = label_response.text.strip().split("\n")

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, bw, bh = map(float, parts)
                        xmin = (x - bw / 2) * width
                        ymin = (y - bh / 2) * height
                        xmax = (x + bw / 2) * width
                        ymax = (y + bh / 2) * height
                        draw.rectangle(
                            [xmin, ymin, xmax, ymax],
                            outline="red",
                            width=2,
                        )

                with img_cols[i % 3]:
                    st.image(
                        draw_img,
                        caption=path.split("/")[-1],
                        width=140,  # small thumbnail
                    )
            except Exception as exc:
                st.error(f"Erreur sur l'image {path} : {exc}")
    else:
        st.info("Aucune image trouvée pour le dataset de détection.")
