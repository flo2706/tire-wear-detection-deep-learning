<h1 align="center">JET – Jedha Evaluation Tyres</h1>
<h3 align="center">Projet Final – Deep Learning & Computer Vision</h3>

<p align="center"><em>Détection automatique de pneus + classification de l’usure</em></p>
<br>

---

## Objectif du projet

Chaque année, l’usure des pneus est responsable de nombreux accidents.  
Le contrôle reste manuel, irrégulier et peu fiable.

**Objectif :** créer une solution de Computer Vision capable de :
1. **Détecter automatiquement les pneus** dans une image (YOLOv8)
2. **Évaluer leur état (Bon / Usé)** via un modèle MobileNetV2 finetuné
3. Fournir une **application web utilisable par tous**

---

## Pipeline Data & Deep Learning

### **1. Détection – YOLOv8m**
- Dataset Roboflow (annotations YOLO)
- 1 classe : pneu  
- Scores :
  - mAP@50 ≈ **97%**
  - Precision ≈ 94% / Recall ≈ 92%

Sert à isoler automatiquement le pneu avant classification.

---

### **2. Classification – MobileNetV2 finetuné**
- Transfer learning + fine tuning des 10 dernières couches  
- Adam (lr = 1e-5), BinaryCrossentropy  
- Meilleur modèle à l’epoch 23 → val_accuracy ≈ **0.86**

 Modèle léger et optimisé pour le déploiement web.

*(Une baseline InceptionV3 a été utilisée pour référence.)*

---

## Données

### **Dataset Classification**
- Kaggle : Tire Quality Classification  
- 2 classes : `good` / `defective`

### **Dataset Détection**
- Roboflow : images + labels YOLO

Datasets publics & anonymes → conformes RGPD.

---

## Pipeline prédictif complet

1. Upload image (pneu ou véhicule)
2. Détection des pneus via **YOLOv8**
3. Recadrage + preprocessing (OpenCV + MobileNetV2)
4. Classification **Bon / Usé**
5. Découpage 4×4 pour analyse locale (zones 🟩/🟥)
6. Affichage final dans l’UI Streamlit

---

## Application Streamlit (Hugging Face Spaces)

- **Overview** : présentation et contexte  
- **Dataset** : exploration des données  
- **Model** : courbes, matrices de confusion, détails techniques  
- **Predictions** : upload + détection + analyse complète du pneu  

Application pensée pour un **public non technique**.

---

## Résultats clés

| Task                 | Modèle        | Score |
|---------------------|---------------|--------|
| Détection           | YOLOv8m       | mAP@50 ≈ **97%** |
| Classification       | MobileNetV2   | Val_acc ≈ **0.86** |
| Analyse locale      | 4×4 zones     | Cohérence visuelle |

---

## Stack technique

- **DL** : TensorFlow / Keras (InceptionV3, MobileNetV2)  
- **Object Detection** : YOLOv8 (Ultralytics)  
- **CV** : OpenCV, Pillow  
- **Data** : pandas, numpy, scikit-image  
- **Visualization** : matplotlib, plotly  
- **App** : Streamlit  
- **Déploiement** : Hugging Face Spaces  
- **Hub modèles** : Hugging Face  

---

## Améliorations possibles

- Heatmaps avancées (Grad-CAM)
- Détection d'autres défauts (hernie, craquelures…)
- Version mobile (Android/iOS)
- Pipeline MLOps (monitoring + réentraînement)
- Intégration industrielle (capteurs automatiques)

---

## Projet Certification Jedha

Ce projet démontre :

- la traduction d’un **problème métier** en pipeline ML,
- la maîtrise de la **Computer Vision moderne**,
- la gestion d’un projet **end-to-end** (data → modèle → app → déploiement),
- la capacité à **vulgariser** les choix techniques.

---
