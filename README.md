<h1 align="center">Tire Wear Detection - Deep Learning & Computer Vision</h1>
<h3 align="center">JET - Jedha Evaluation Tyre</h3>
<p align="center"><em>Détection automatique de pneus + classification de l’usure</em></p>
<br>

---

## Objectif du projet

L’usure des pneus reste un facteur majeur d’accidents, principalement en raison d’un contrôle manuel, irrégulier et subjectif.

L’objectif de ce projet est de concevoir une solution complète de Computer Vision capable de :
1. **Détecter automatiquement les pneus** dans une image (YOLOv8)
2. **Évaluer leur état (apte / inapte à la conduite)** (MobileNetV2 finetuné)
3. Offrir une **application web simple d’usage**, utilisable par tout public

---

## Pipeline Data & Deep Learning

### **1. Détection – YOLOv8m**
- 1 classe : pneu  
- Scores :
  - mAP@50 ≈ **97%**
  - Precision ≈ 93%
  - Recall ≈ 94%

Utilisé pour isoler automatiquement chaque pneu avant classification.

---

### **2. Classification – MobileNetV2 finetuné**
- Transfer learning + fine tuning (10 dernières couches)  
- Optimiseur : Adam (lr = 1e-5)
- Loss : BinaryCrossentropy  
- Val_accuracy ≈ **0.86**

*(Une baseline InceptionV3 a été utilisée pour référence.)*

---

## Données

### **Dataset Classification**
- Source originale : Kaggle : Tire Quality Classification
- Dataset utilisé : 
- 2 classes : `good` / `defective`

### **Dataset Détection**
- Source originale : Roboflow : images annotées (format YOLOv8)
- Dataset utilisé : 

Les données sont publiques, anonymes et conformes aux principes du RGPD.

---

## Pipeline prédictif 

1. Import de l’image (voiture ou pneu isolé)
2. Détection des pneus avec **YOLOv8**
3. Recadrage + preprocessing OpenCV
4. Classification globale Bon / Usé avec MobileNetV2
5. Analyse locale 4×4 :
  - zones 🟩 = bonnes
  - zones 🟥 = usées
6. Affichage détaillé dans l’UI Streamlit

---

## Application Streamlit (Hugging Face Spaces)
Ce projet inclut une application complète, accessible en ligne :

👉 https://huggingface.co/spaces/jedhajet/jedhaJeTter

Sections de l’application :
- **Overview** : contexte & objectifs 
- **Dataset** : inspection des données  
- **Model** : performances, courbes, matrices de confusion 
- **Predictions** : analyse automatique d’image 

Conçue pour un public non expert, avec une interface pédagogique.

---

## Résultats 

| Tâche             | Modèle        | Score |
|---------------------|---------------|--------|
| Détection           | YOLOv8m       | mAP@50 ≈ **97%** |
| Classification       | MobileNetV2   | Val_acc ≈ **0.86** |
| Analyse locale      | Grid 4×4      | Cohérence visuelle |

---

## Stack technique

- **Deep Learning** : TensorFlow / Keras (InceptionV3, MobileNetV2)  
- **Object Detection** : YOLOv8 (Ultralytics)  
- **CV** : OpenCV, Pillow  
- **Data** : pandas, numpy, scikit-image  
- **Visualization** : matplotlib, plotly  
- **App** : Streamlit  
- **Déploiement** : Hugging Face Spaces  
- **Hub modèles** : Hugging Face  

---

## Améliorations possibles

- Segmentation précise du **pneu uniquement** (exclusion de la jante), non implémentée dans ce projet faute de **données annotées adaptées**, mais identifiée comme un levier clé d’amélioration
- Heatmaps avancées (Grad-CAM)
- Détection d'autres défauts (hernie, craquelures…)
- Version mobile (Android/iOS)
- Pipeline MLOps (monitoring + réentraînement)
- Système embarqué industriel (capteurs automatiques)

---

## Cadre du projet

Ce projet démontre :

- la capacité à transformer un **problème métier** en **solution de deep learning complète**
- la conception d’un **pipeline end-to-end** : data → modèle → application → déploiement
- la capacité à **expliquer et vulgariser** des choix techniques auprès d’un public non expert
- la capacité à **collaborer efficacement en binôme** sur un projet data, de la conception au déploiement

---

## Contexte

Projet final réalisé dans le cadre de la certification  
**« Concepteur Développeur en Sciences des Données » (RNCP 35288 – Jedha)**,
en collaboration avec **Youenn Patat**.
