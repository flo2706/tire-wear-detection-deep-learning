import streamlit as st

# ---------------------- MAIN LANDING PAGE ----------------------
# This page introduces the project, context and navigation to other pages.

# Main title / subtitle (HTML styled)
st.markdown(
    """
    <div style="text-align: center; font-size: 1.5rem; color: gray;">
        Jedha Evaluation Tyres -test
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align: center; font-size: 1.1rem; color: gray;'>A Convolutional Neural Network Project</div>",
    unsafe_allow_html=True,
)

st.write("")

# Centered logo
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("logo/Logo_JET.png", use_container_width=True)

st.write("")

# Project description 
st.markdown("""
L'objectif de ce projet est de déployer un modèle de *deep learning* permettant de contrôler la qualité des pneus à partir d'une image importée.  
La classification est la suivante :

🛞 Contrôle de la qualité du pneu par image :  
* Bon état (apte à rouler) ✅  
* Mauvais état (pas apte à rouler / à changer) ❌  

Les informations sur les données utilisées se trouvent dans la page `Dataset`.  
Les différentes informations sur le modèle de baseline, le plus simple et correct pour des premiers résultats,
ainsi que sur le meilleur modèle obtenu pour effectuer les prédictions se trouvent sur la page `Model`.  
La page `Predictions` vous permet de charger une photo de pneu ou de voiture et d'obtenir une prédiction sur sa qualité, avec le taux de confiance du modèle.

Enfin, pour un usage futur à plus grande échelle, on pourrait imaginer la mise en place de capteurs industrialisés 
placés en face de chaque roue. À chaque démarrage du véhicule, le capteur prendrait une photo du pneu.  
Avec le `JET model` implémenté à l'intérieur, il calculerait l'état du pneu et pourrait renvoyer l'information
au conducteur sur le tableau de bord, indiquant si tout va bien ✅ ou s'il y a un danger ❌ et un changement à faire.
""")
