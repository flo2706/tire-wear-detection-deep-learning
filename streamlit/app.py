import streamlit as st
from PIL import Image

# Config
page_icon_path="logo/logo_app_favicon.png"
st.set_page_config(page_title="JET", page_icon=page_icon_path, layout="wide")

with st.sidebar:
    st.image("logo/Logo_JET.png", use_container_width=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <h3><b>JET</b></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: center; font-size: 0.875rem; color: gray;">
            Détection d’usure de pneu par IA
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()


predictions= st.Page("pages/predictions.py", title="Predictions")
main_page = st.Page("pages/main_page.py", title="Overview")
dataset = st.Page("pages/dataset.py", title="Dataset")
model = st.Page("pages/model.py", title="Model")


# Set up navigation
pg = st.navigation([predictions, main_page, dataset, model])

# Run the selected page
pg.run()


