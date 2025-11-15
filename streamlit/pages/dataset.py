# import streamlit as st
# import streamlit.components.v1 as components
# from huggingface_hub import list_repo_files
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import random
# import requests
# from io import BytesIO
# import os
# from PIL import Image
# import pandas as pd
# import numpy as np
# import requests
# import plotly.express as px
# import seaborn as sns
# from skimage.color import rgb2lab
# import asyncio
# import aiohttp

# components.html(
#     """
#     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
#     <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
#     <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>

#     <style>
#       .card {
#         border-radius: 12px;
#         margin-bottom: 15px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#       }
#       .btn-link {
#         font-size: 1.05rem;
#         color: #444;
#         text-decoration: none;
#       }
#       .btn-link:hover {
#         color: #111;
#       }
#       pre {
#         background-color: #f8f9fa;
#         padding: 15px;
#         border-radius: 8px;
#         font-size: 0.85rem;
#       }
#       ul { padding-left: 1.2rem; }
#     </style>

#     <div class="container mt-4">
#       <div class="row">
#         <!-- Classification Dataset -->
#         <div class="col-md-6">
#           <div id="accordion1">
#             <div class="card">
#               <div class="card-header" id="head1">
#                 <h5 class="mb-0">
#                   <button class="btn btn-link" data-toggle="collapse" data-target="#collapse1" aria-expanded="true">
#                     🗂️ Dataset 1 : Classification (Inception_v3 / Kaggle)
#                   </button>
#                 </h5>
#               </div>
#               <div id="collapse1" class="collapse show" data-parent="#accordion1">
#                 <div class="card-body">
#                   <ul>
#                     <li><strong>Lien Kaggle :</strong> <a href="https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data" target="_blank">Accéder</a></li>
#                     <li><strong>Lien HuggingFace :</strong> <a href="https://huggingface.co/datasets/flodussart/tires_project_roboflow" target="_blank">Accéder</a></li>
#                     <li><strong>Auteur :</strong> Chirag CHAUHAN</li>
#                     <li><strong>Mis à jour :</strong> Il y a 2 ans</li>
#                     <li><strong>Licence :</strong> <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a></li>
#                   </ul>
#                   <p>Ce dataset contient 1854 images réparties en deux classes : <strong>good</strong> et <strong>defective</strong>.</p>
#                   <p>Utile pour entraîner des modèles de classification d’images pour la sécurité automobile.</p>

#                   <pre>
# Images numériques de pneus/
# ├── defective/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# └── good/
#     ├── img1.jpg
#     ├── img2.jpg
#     └── ...
#                   </pre>
#                 </div>
#               </div>
#             </div>
#           </div>
#         </div>

#         <!-- Detection Dataset -->
#         <div class="col-md-6">
#           <div id="accordion2">
#             <div class="card">
#               <div class="card-header" id="head2">
#                 <h5 class="mb-0">
#                   <button class="btn btn-link" data-toggle="collapse" data-target="#collapse2" aria-expanded="true">
#                     📦 Dataset 2 : Détection (YOLOv8 / Roboflow)
#                   </button>
#                 </h5>
#               </div>
#               <div id="collapse2" class="collapse show" data-parent="#accordion2">
#                 <div class="card-body">
#                   <ul>
#                     <li><strong>Lien Roboflow :</strong> <a href="https://universe.roboflow.com/iotml/tire-dataset/dataset/2" target="_blank">Accéder</a></li>
#                     <li><strong>Lien HuggingFace: </strong> <a href=https://huggingface.co/datasets/flodussart/tires_project" target="_blank">Accéder</a></li>
#                     <li><strong>Titre :</strong> Tire Dataset – Computer Vision Project</li>
#                     <li><strong>Année :</strong> 2022</li>
#                   </ul>
#                   <p>Ce dataset est formaté pour YOLOv8 avec annotations pour la <strong>détection d’objets</strong>. Il contient 1464 images de train, 191 images de val et 104 images de test.</p>

#                   <pre>
# Dataset de détection YOLOv8/
# ├── train/
# │   ├── images/
# │   └── labels/
# ├── valid/
# │   ├── images/
# │   └── labels/
# ├── test/
# │   ├── images/
# │   └── labels/
# ├── data.yaml
# ├── README.dataset.txt
# └── README.roboflow.txt
#                   </pre>
#                 </div>
#               </div>
#             </div>
#           </div>
#         </div>
#       </div>
#     </div>
#     """,
#     height=900
# )


# # Init session state for refresh buttons
# if "refresh" not in st.session_state:
#     st.session_state.refresh = 0

# # Custom CSS for smaller text and centered headers
# st.markdown("""
#      <style>
#         h1, h2, h3, h4, h5, h6 { text-align: center; }
#         .markdown-text-container { font-size: 0.85rem !important; }
#         .element-container img + div {
#             font-size: 0.75rem !important;
#             text-align: center;
#             color: #555;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # ---- ASYNC IMAGE FETCHING ----
# async def fetch_image(session, url):
#     try:
#         async with session.get(url) as response:
#             return await response.read()
#     except:
#         return None

# async def fetch_images_async(urls):
#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch_image(session, url) for url in urls]
#         return await asyncio.gather(*tasks)

# # ---- CONFIG ----
# REPO_ID = "flodussart/tires_project"
# BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"

# @st.cache_data
# def get_image_df():
#     files = list_repo_files(REPO_ID, repo_type="dataset")
#     image_paths = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and ("good/" in f or "defective/" in f)]
#     data = []
#     for path in image_paths:
#         label = "good" if "good/" in path else "defective"
#         url = BASE_URL + path
#         data.append({"url": url, "label": label})
#     return pd.DataFrame(data)

# # ---- RGB + LAB FEATURE EXTRACTION ----
# def compute_color_features(df, sample_per_class=100):
#     data = []
#     for label in ["good", "defective"]:
#         subset = df[df['label'] == label].sample(min(sample_per_class, len(df[df['label'] == label])), random_state=42)
#         urls = subset['url'].tolist()
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         images_bytes = loop.run_until_complete(fetch_images_async(urls))
#         for img_bytes, url in zip(images_bytes, urls):
#             try:
#                 img = Image.open(BytesIO(img_bytes)).convert('RGB').resize((64, 64))
#                 arr = np.array(img)
#                 lab = rgb2lab(arr / 255.0)
#                 data.append({
#                     "url": url,
#                     "label": label,
#                     "R": arr[..., 0].mean(),
#                     "G": arr[..., 1].mean(),
#                     "B": arr[..., 2].mean(),
#                     "L": lab[..., 0].mean(),
#                     "a": lab[..., 1].mean(),
#                     "b": lab[..., 2].mean(),
#                 })
#             except:
#                 continue
#     return pd.DataFrame(data)


# # ---- DATASET CLASSIFICATION ----
# st.markdown("---")
# with st.expander("🗂️ Dataset 1 – Classification (Inception_v3/Kaggle)", expanded=False):

#     st.markdown("### Aperçu du dataset de classification")

#     df = get_image_df()  
#     sample_size = st.slider("Nombre d'images analysées (par classe)", 10, 200, 50)

#     rgb_df = compute_color_features(df, sample_per_class=sample_size)

#     # Quick image preview
#     sampled = df.sample(min(9, len(df)))
#     fig, ax = plt.subplots(3, 3, figsize=(6, 6))
#     for i, row in enumerate(sampled.itertuples()):
#         response = requests.get(row.url)
#         img = mpimg.imread(BytesIO(response.content), format='jpg')
#         ax[i // 3, i % 3].imshow(img)
#         ax[i // 3, i % 3].set_title(row.label, fontsize=8)
#         ax[i // 3, i % 3].axis("off")
#     st.pyplot(fig)

#     # Statistiques 
#     st.markdown("### Statistiques rapides")
#     colA, colB = st.columns(2)
#     with colA:
#         label_counts = df['label'].value_counts().reset_index()
#         label_counts.columns = ['label', 'count']
#         fig_bar = px.bar(label_counts, x='label', y='count', color='label', text='count',
#                          color_discrete_sequence=['gray', 'dimgray'])
#         fig_bar.update_layout(yaxis_title="Nombre d'images", xaxis_title="Label", showlegend=False, height=300)
#         st.plotly_chart(fig_bar, use_container_width=True)

#     with colB:
#         st.metric("Total d'images", len(df))
#         st.metric("Nombre de classes", df['label'].nunique())

#     # Filter by class
#     st.markdown("### Affichage filtré par classe")
#     selected_label = st.selectbox("Choisir une classe :", df['label'].unique())
#     subset = df[df['label'] == selected_label].sample(n=min(3, len(df[df['label'] == selected_label])), random_state=1)
#     img_cols = st.columns(3)
#     for i, row in enumerate(subset.itertuples()):
#         with img_cols[i % 3]:
#             st.image(row.url, use_container_width=True)

# # --- CONFIG 2e dataset ---
# # ---- DATASET DÉTECTION (YOLOv8) ----
# st.markdown("---")
# with st.expander("📦 Dataset 2 – Détection (YOLOv8 / Roboflow)", expanded=True):

#     from PIL import ImageDraw

#     DETECT_REPO_ID = "flodussart/tires_project_roboflow"
#     st.markdown("### Visualisation des annotations")

#     @st.cache_data
#     def get_yolo_image_paths():
#         files = list_repo_files(DETECT_REPO_ID, repo_type="dataset")
#         return [f for f in files if f.startswith("train/images/") and f.endswith(('.jpg', '.jpeg', '.png'))]

#     image_paths = get_yolo_image_paths()
#     sample_paths = random.sample(image_paths, min(6, len(image_paths)))

#     img_cols = st.columns(3)  

#     for i, path in enumerate(sample_paths):
#         image_url = f"https://huggingface.co/datasets/{DETECT_REPO_ID}/resolve/main/{path}"
#         label_path = path.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
#         label_url = f"https://huggingface.co/datasets/{DETECT_REPO_ID}/resolve/main/{label_path}"

#         try:
#             # load img
#             img_response = requests.get(image_url)
#             img = Image.open(BytesIO(img_response.content)).convert("RGB")
#             w, h = img.size

#             # Image annotation
#             draw_img = img.copy()
#             draw = ImageDraw.Draw(draw_img)

#             label_response = requests.get(label_url)
#             label_lines = label_response.text.strip().split("\n")

#             for line in label_lines:
#                 parts = line.strip().split()
#                 if len(parts) == 5:
#                     cls, x, y, bw, bh = map(float, parts)
#                     xmin = (x - bw / 2) * w
#                     ymin = (y - bh / 2) * h
#                     xmax = (x + bw / 2) * w
#                     ymax = (y + bh / 2) * h
#                     draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
#                     draw.text((xmin, ymin), f"Class {int(cls)}", fill="red")

#             with img_cols[i % 3]:
#                 st.image(draw_img, caption=path.split("/")[-1], width=250)

#         except Exception as e:
#             st.error(f"Erreur image {path} : {e}")
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import list_repo_files
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import plotly.express as px
from skimage.color import rgb2lab

# ---------------------- HEADER HTML (Bootstrap + description) ----------------------

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.bundle.min.js"></script>

    <style>
      .card {
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
      }
      .btn-link {
        font-size: 1.05rem;
        color: #444;
        text-decoration: none;
      }
      .btn-link:hover {
        color: #111;
      }
      pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        font-size: 0.85rem;
      }
      ul { padding-left: 1.2rem; }
    </style>

    <div class="container mt-4">
      <div class="row">
        <!-- Classification Dataset -->
        <div class="col-md-6">
          <div id="accordion1">
            <div class="card">
              <div class="card-header" id="head1">
                <h5 class="mb-0">
                  <button class="btn btn-link" data-toggle="collapse" data-target="#collapse1" aria-expanded="true">
                    🗂️ Dataset 1 : Classification (Inception_v3 / Kaggle)
                  </button>
                </h5>
              </div>
              <div id="collapse1" class="collapse show" data-parent="#accordion1">
                <div class="card-body">
                  <ul>
                    <li><strong>Source Kaggle :</strong> <a href="https://www.kaggle.com/datasets/warcoder/tyre-quality-classification/data" target="_blank">Accéder</a></li>
                    <li><strong>Version HuggingFace :</strong> <a href="https://huggingface.co/datasets/flodussart/tires_project_roboflow" target="_blank">Accéder</a></li>
                    <li><strong>Auteur :</strong> Chirag CHAUHAN</li>
                    <li><strong>Licence :</strong> <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a></li>
                  </ul>
                  <p>Ce dataset contient des images réparties en deux classes : <strong>good</strong> et <strong>defective</strong>.
                  Il est utilisé pour entraîner un modèle de classification d’usure de pneu.</p>

                  <pre>
Images numériques de pneus/
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
            <div class="card">
              <div class="card-header" id="head2">
                <h5 class="mb-0">
                  <button class="btn btn-link" data-toggle="collapse" data-target="#collapse2" aria-expanded="true">
                    📦 Dataset 2 : Détection (YOLOv8 / Roboflow)
                  </button>
                </h5>
              </div>
              <div id="collapse2" class="collapse show" data-parent="#accordion2">
                <div class="card-body">
                  <ul>
                    <li><strong>Source Roboflow :</strong> <a href="https://universe.roboflow.com/iotml/tire-dataset/dataset/2" target="_blank">Accéder</a></li>
                    <li><strong>Version HuggingFace :</strong> <a href="https://huggingface.co/datasets/flodussart/tires_project" target="_blank">Accéder</a></li>
                    <li><strong>Titre :</strong> Tire Dataset – Computer Vision Project</li>
                  </ul>
                  <p>Ce dataset est formaté pour YOLOv8 avec annotations pour la <strong>détection d’objets</strong>.
                  Il contient des images de train / validation / test avec fichiers d’annotations au format YOLO.</p>

                  <pre>
Dataset de détection YOLOv8/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
├── README.dataset.txt
└── README.roboflow.txt
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """,
    height=900,
)

# ---------------------- GLOBAL STYLES ----------------------

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

# ---------------------- DATASET CONFIG ----------------------

CLASSIF_REPO_ID = "flodussart/tires_project"
CLASSIF_BASE_URL = f"https://huggingface.co/datasets/{CLASSIF_REPO_ID}/resolve/main/"

DETECT_REPO_ID = "flodussart/tires_project_roboflow"
DETECT_BASE_URL = f"https://huggingface.co/datasets/{DETECT_REPO_ID}/resolve/main/"

# ---------------------- HELPERS: FILE LOADING ----------------------

@st.cache_data
def get_classification_image_df():
    """List classification images (good / defective) from the HF dataset."""
    files = list_repo_files(CLASSIF_REPO_ID, repo_type="dataset")
    image_paths = [
        f
        for f in files
        if f.endswith((".jpg", ".jpeg", ".png"))
        and ("good/" in f or "defective/" in f)
    ]
    data = []
    for path in image_paths:
        label = "good" if "good/" in path else "defective"
        url = CLASSIF_BASE_URL + path
        data.append({"url": url, "label": label})
    return pd.DataFrame(data)

def compute_color_features(df: pd.DataFrame, sample_per_class: int = 100) -> pd.DataFrame:
    """Compute simple color features (RGB + Lab) from a sample of images per class."""
    data = []
    for label in ["good", "defective"]:
        class_df = df[df["label"] == label]
        if len(class_df) == 0:
            continue

        n = min(sample_per_class, len(class_df))
        subset = class_df.sample(n=n, random_state=42)

        for _, row in subset.iterrows():
            try:
                resp = requests.get(row["url"])
                img = Image.open(BytesIO(resp.content)).convert("RGB").resize((64, 64))
                arr = np.array(img)
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
                continue

    return pd.DataFrame(data)

@st.cache_data
def get_detection_image_paths():
    """List YOLO detection images (train/images) from the HF dataset."""
    files = list_repo_files(DETECT_REPO_ID, repo_type="dataset")
    return [
        f
        for f in files
        if f.startswith("train/images/")
        and f.endswith((".jpg", ".jpeg", ".png"))
    ]

# ---------------------- DATASET 1 : CLASSIFICATION ----------------------

st.markdown("---")
with st.expander("🗂️ Dataset 1 – Classification (Inception_v3 / Kaggle)", expanded=False):

    st.markdown("### Aperçu du dataset de classification")

    df = get_classification_image_df()

    # Slider for number of images used in color feature analysis
    sample_size = st.slider(
        "Nombre d'images analysées (par classe)",
        10,
        200,
        50,
        step=10,
    )

    rgb_df = compute_color_features(df, sample_per_class=sample_size)

    # Preview of a small sample of images
    st.markdown("#### Aperçu d'un échantillon d'images")
    sampled = df.sample(min(9, len(df)), random_state=1)
    fig, ax = plt.subplots(3, 3, figsize=(6, 6))
    for i, row in enumerate(sampled.itertuples()):
        resp = requests.get(row.url)
        img = mpimg.imread(BytesIO(resp.content), format="jpg")
        ax[i // 3, i % 3].imshow(img)
        ax[i // 3, i % 3].set_title(row.label, fontsize=8)
        ax[i // 3, i % 3].axis("off")
    st.pyplot(fig)

    # Quick global statistics
    st.markdown("### Statistiques rapides")
    colA, colB = st.columns(2)

    with colA:
        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        fig_bar = px.bar(
            label_counts,
            x="label",
            y="count",
            color="label",
            text="count",
            color_discrete_sequence=["gray", "dimgray"],
        )
        fig_bar.update_layout(
            yaxis_title="Nombre d'images",
            xaxis_title="Label",
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with colB:
        st.metric("Total d'images", len(df))
        st.metric("Nombre de classes", df["label"].nunique())

    # Filtered display by class
    st.markdown("### Affichage filtré par classe")
    selected_label = st.selectbox("Choisir une classe :", df["label"].unique())
    subset = df[df["label"] == selected_label].sample(
        n=min(3, len(df[df["label"] == selected_label])),
        random_state=1,
    )
    img_cols = st.columns(3)
    for i, row in enumerate(subset.itertuples()):
        with img_cols[i % 3]:
            st.image(row.url, use_container_width=True)

# ---------------------- DATASET 2 : DÉTECTION (YOLOv8) ----------------------

st.markdown("---")
with st.expander("📦 Dataset 2 – Détection (YOLOv8 / Roboflow)", expanded=True):

    st.markdown("### Visualisation de quelques annotations YOLOv8")

    image_paths = get_detection_image_paths()
    if len(image_paths) == 0:
        st.warning("Aucune image trouvée dans le dataset de détection.")
    else:
        sample_paths = random.sample(image_paths, min(6, len(image_paths)))
        img_cols = st.columns(3)

        for i, path in enumerate(sample_paths):
            image_url = f"{DETECT_BASE_URL}{path}"
            label_path = path.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
            label_url = f"{DETECT_BASE_URL}{label_path}"

            try:
                # Load image
                img_response = requests.get(image_url)
                img = Image.open(BytesIO(img_response.content)).convert("RGB")
                w, h = img.size

                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)

                # Load YOLO labels
                label_response = requests.get(label_url)
                lines = label_response.text.strip().split("\n")

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, bw, bh = map(float, parts)
                        xmin = (x - bw / 2) * w
                        ymin = (y - bh / 2) * h
                        xmax = (x + bw / 2) * w
                        ymax = (y + bh / 2) * h
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
                        draw.text((xmin, ymin), f"class {int(cls)}", fill="red")

                with img_cols[i % 3]:
                    st.image(
                        draw_img,
                        caption=path.split("/")[-1],
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Erreur sur l'image {path} : {e}")
