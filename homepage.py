import streamlit as st

# Set page title and favicon.
st.set_page_config(
    page_title="Page de sélection d'outils",
    page_icon="🤖",
)

st.title("🤖 Outils Machine Learning CDGD3")

with st.expander("Informations sur l'outil de Machine Learning"):
    st.markdown('**Que peut-on faire avec cette application ?**')
    st.info("Cette application permet de lancer des outils de Machine Learning pour l'analyse de données à CDGD3. Vous pouvez choisir un outil dans la liste ci-dessous.")

    st.markdown('**Comment utiliser cette application ?**')
    st.markdown('1. **Choisissez un outil dans la liste ci-contre.**')
    st.markdown('2. **Suivez les instructions pour utiliser l\'outil. Il faudra importer un fichier de données pour chaque outil.**')
    st.markdown('3. **Consultez les résultats et les graphiques générés.**')


    st.markdown('Librairies utilisées:')
    st.code('''- Pandas pour la manipulation des données
    - Scikit-learn pour la construction du modèle
    - XGBoost pour la prédiction des taux de remplissage  
    - Altair pour la visualisation des données
    - Streamlit pour la création de l\'application web 
    ''', language='markdown')

st.sidebar.success("Choisissez un outil dans la liste ci-dessous.")

