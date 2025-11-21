import streamlit as st

# ==============================================================
#                    MAIN LANDING PAGE (HOME)
#            Intro page with project description + navigation
# ==============================================================


# ---------------------- TITLE ----------------------
st.markdown(
    """
    <div style="text-align: center; font-size: 3rem; color: gray;">
        Jedha Evaluation Tyres
    </div>
    """,
    unsafe_allow_html=True,
)

# Subtitle 
st.markdown(
    """
    <div style='text-align: center; font-size: 2rem; color: gray;'>
        A Convolutional Neural Network Project
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ---------------------- PROJECT DESCRIPTION ----------------------
st.markdown(
    """
### Objectif du projet

Ce projet vise à déployer un modèle de *deep learning* capable d'évaluer automatiquement
l’état d’un pneu à partir d’une image importée.

Chaque pneu est classé dans l’une des deux catégories suivantes :

- **Bon état** – apte à la conduite ✅  
- **Mauvais état** – inapte à la conduite / dangereux ❌  

---

### Navigation dans le dashboard

**Pages informatives**
- **Dataset** : description et exploration des données utilisées pour l'entraînement des modèles  
- **Model** : présentation des modèles de classification et de détection  

**Page de prédiction**
- **Predictions** : téléversez une image (pneu seul ou véhicule) pour obtenir une prédiction
  accompagnée d’un score de confiance sur l’état du pneu.

---

### Contexte et perspectives

Dans une perspective de déploiement à grande échelle, on peut envisager l’intégration de capteurs
industrialisés positionnés face à chaque roue.  
À chaque démarrage du véhicule, une photo du pneu serait prise automatiquement.

Le modèle `JET` embarqué analyserait alors l’image en temps réel et renverrait l’information au conducteur
sur le tableau de bord :

- ✅ **Tout est en ordre**  
- ❌ **Anomalie détectée – contrôle ou remplacement recommandé**

Cette approche contribuerait à renforcer la sécurité routière et à faciliter la maintenance prédictive des véhicules.
"""
)
