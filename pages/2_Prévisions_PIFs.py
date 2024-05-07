import itertools
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import altair as alt
# import time
import zipfile
import datetime
from functools import reduce
from copy import deepcopy
from pykalman import KalmanFilter, UnscentedKalmanFilter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, TimeSeriesSplit
import xgboost as xgb
from itertools import product
import ast

from stqdm import stqdm

import joblib

# import streamlit_extras as stx
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_extras.mandatory_date_range import date_range_picker






# Page title
st.set_page_config(page_title='CDGD3 Taux Présentation', 
                   page_icon='🤖', layout="wide", initial_sidebar_state="expanded", 
                   menu_items={"Report a bug": "mailto:vincent.nazzareno@adp.fr",
                               "About" : "Application développée par Vincent Nazzareno, Machine Learning Engineer Intern à CDGD3"}
)

st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 550px;
           max-width: 550px;
       }
       """,
        unsafe_allow_html=True,
    )   


st.title('🤖 Machine Learning CDGD3 🤖')

with st.expander("Informations sur l'application"):
    st.markdown('**Que peut-on faire avec cette application ?**')
    st.info("Cette application permet de construire un modèle de machine learning pour prédire l'affluence aux PIFs sur une période donnée. Les données sont issues du datamart de CDGD3.")

    st.markdown('**Comment utiliser cette application ?**')
    st.markdown('1. **Charger un fichier CSV** : Chargez un fichier CSV contenant les données à analyser')
    st.markdown('2. **Paramètres** : Réglez les paramètres du modèle.')
    st.markdown("3. **Exécution** : L'éxécution de la prédiction se fait automatiquement après le chargement des données et le réglage des paramètres.")
    st.markdown('4. **Résultats** : Consultez les résultats sur la période.')
    st.markdown('5. **Téléchargement** : Téléchargez les prédictions et les données traitées.')

    st.markdown('Librairies utilisées:')
    st.code('''- Pandas pour la manipulation des données
    - Scikit-learn pour la construction du modèle
    - XGBoost pour la prédiction des taux de remplissage  
    - Altair pour la visualisation des données
    - Streamlit pour la création de l\'application web 
    ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:

    st.header("1. Paramètres de l'application")
    sequence_length = st.slider('Horizons de prédiction', 10, 20, 10, 10)
    smoothnes_factor = st.slider('Facteur de lissage', 2, 15, 3, 1)
    sleep_time = st.slider('Sleep time', 0, 3, 1)
    compute_offre_matrix = st.checkbox('Calculer la matrice d\'offre par compagnie et sous-type d\'avion', value=True)

    # Load data
    st.header('2.1. Fichier Source')

    st.markdown('**1. Données Taux Remplissage**')
    separator_presentation = st.selectbox('Séparateur données datamart (par défaut tabulation)', [';', ',', '\t', '|'], index=0)
    # col_to_input = st.multiselect('Colonnes à importer', ['Source', 'Jour', 'Plage', 'Type de mouvement', 'Code IATA compagnie',
    #                                                         'Numéro de vol', 'Immatriculation', 'Sous-type avion', 'Ville',
    #                                                         'Code aéroport IATA', 'Faisceau géographique',
    #                                                         'Nombre de passagers réalisés', 'Nombre de sièges offerts'], 
    #                                                         default=['Source', 'Jour', 'Plage', 'Type de mouvement', 'Code IATA compagnie',
    #                                                         'Numéro de vol', 'Immatriculation', 'Sous-type avion', 'Ville',
    #                                                         'Code aéroport IATA', 'Faisceau géographique',
    #                                                         'Nombre de passagers réalisés', 'Nombre de sièges offerts'])
    
    col_unique_id_group_y = st.multiselect('Colonnes pour le groupement', ['Source', 'Local Date', 'Plage', 'A/D', 'Cie Ope', 'Num Vol',
                                                                            'Sous-type avion', 'Ville', 'Prov Dest', 'Faisceau géographique',
                                                                            'PAX TOT', 'Offre', 'is_ferie', 'nom_jour_ferie', 'Zone A', 'Zone B',
                                                                            'Zone C', 'Zone A vacance', 'Zone B vacance', 'Zone C vacance', 'Year',
                                                                            'Taux Remplissage', 'Jour de la semaine', 'PAX TOT SHIFT',
                                                                            'Taux Remplissage SHIFT', 'id', 'semaine_sin_time', 'semaine_cos_time',
                                                                            'month_sin_time', 'month_cos_time'], 
                                                            default=['Local Date', 'Cie Ope', 'Prov Dest', 'Plage'])


    class CustomXGBRegressor(xgb.XGBRegressor):
        def _objective(self, y_pred, y_true):
            
            mse = np.mean(np.square(y_pred - y_true))
            temporal_distance = np.mean(np.abs(np.argmax(y_pred, axis=1) - np.argmax(y_true, axis=1)))
            total_loss = mse + 0.2 * temporal_distance
            
            # return mse, temporal_distance
            return total_loss
        
    uploaded_file_y = st.file_uploader("Fichier Programme Concat", type=["csv", "txt"])
    uploaded_model_presentation = st.file_uploader("Modèle entrainé Présentation", type=["json"])
    uploaded_model_repartition = st.file_uploader("Modèle entrainé Répartition", type=["json"])
    download_final_df = False

    
    if uploaded_file_y is not None and download_final_df != True:
        @st.cache_data
        def get_x_previ_df(uploaded_file_y, sep):
            
            df = pd.read_csv(uploaded_file_y, sep=sep, date_format='%d/%m/%Y')
            try:
                df.drop(['Affectation', 'Etat du vol'], axis=1, inplace=True)
            except KeyError:
                st.warning("Les colonnes 'Affectation' et 'Etat du vol' n'ont pas été trouvées dans le fichier source. Essayer un autre séparateur.")

            df['Local Date'] = df['Local Date'].apply(lambda x: x.split(' ')[0])
            df['Porteur'].fillna('MP', inplace=True)
            df['Libellé terminal'] = df['Libellé terminal'].str.replace("T1_Inter","Terminal 1")
            df['Libellé terminal'] = df['Libellé terminal'].str.replace("T1_5","Terminal 1_5")
            df['Libellé terminal'] = df['Libellé terminal'].str.replace("T1_6","Terminal 1_6")
            df['Cie Ope'].dropna(inplace=True)
            df['Num Vol'].dropna(inplace=True)

            # numerical_col = ['Local Date', 'Horaire théorique', 'Pax LOC TOT', 'Pax CNT TOT', 'PAX TOT']
            # categorical_col = ~df.columns.isin(numerical_col)
            df['Local Date'] = df['Local Date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%d/%m/%Y"))

            calendrier_gouv = pd.read_csv(r"ressources\calendar_complete_from_data_gouv.csv", sep=',', date_format='%dd/%mm/%YYYY')
            calendrier_gouv['date'] = calendrier_gouv['date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%d/%m/%Y"))

            calendrier_gouv.fillna('non_ferie', inplace=True)
            df_full = pd.merge(df, calendrier_gouv, left_on='Local Date', right_on='date', how='left')
            df_full.drop(columns=['date'], axis=1, inplace=True)

            df_full['Jour (nb)'] = pd.to_datetime(df_full['Local Date'], format="%d/%m/%Y").dt.dayofweek
            df_full['Semaine'] = pd.to_datetime(df_full['Local Date'], format="%d/%m/%Y").dt.isocalendar().week

            table_faisceau_iata = pd.read_excel(r"ressources\table_faisceau_IATA.xlsx")
            table_faisceau_iata.rename(columns={"Code aéroport IATA":"Prov Dest"}, inplace=True)
            table_faisceau_iata = table_faisceau_iata[['Prov Dest','Faisceau géographique']]
            df_full = df_full.merge(table_faisceau_iata,how='left', left_on='Prov Dest', right_on='Prov Dest')
            df_full['Faisceau géographique'].fillna('Autre Afrique', inplace=True)

            # df_full['Year'] = df_full['Local Date'].dt.year
            # df_full['month'] = df_full['Local Date'].dt.month
            # df_full['day'] = df_full['Local Date'].dt.day
            df_full.drop(columns=['Jour (nb)'], axis=1, inplace=True)
            df_full['Jour de la semaine'] = df_full['Local Date'].apply(lambda x: datetime.datetime.strptime(x,"%d/%m/%Y").weekday())
            df_full['month'] = df_full['Local Date'].apply(lambda x: datetime.datetime.strptime(x,"%d/%m/%Y").month)

            return df_full
        
        data_x = get_x_previ_df(uploaded_file_y, separator_presentation)

        st.dataframe(data_x, height=210, use_container_width=True)
      

# Initiate the model building process
if uploaded_file_y is not None and download_final_df != True and uploaded_model_presentation is not None and uploaded_model_repartition is not None:

    st.warning('La requête a bien été prise en compte, début du traitement.\nNe tentez pas de fermer la fenêtre même si celle-ci semble figée.')
    placeholder = st.empty()
    my_bar = placeholder.progress(5)
    
    st.write('Chargement du modèle ...')
    @st.cache_data
    def load_models():
        # multioutputregressor_presentation = joblib.load(r"ressources\courbe_pres_multiouput_xgb.json") 
        # multioutputregressor_repartition = joblib.load(r"ressources\courbe_rep_multiouput_xgb.json")
        multioutputregressor_presentation = joblib.load(uploaded_model_presentation)
        multioutputregressor_repartition = joblib.load(uploaded_model_repartition)
        
        return multioutputregressor_presentation, multioutputregressor_repartition
    
    multioutputregressor_presentation, multioutputregressor_repartition = load_models()
    
    class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, object_col):
            """Encode the data based on object column provided
            """
            self.object_col = object_col

        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            dummy_df = pd.get_dummies(X[self.object_col], drop_first=True)
            X = X.drop(X[self.object_col], axis=1)
            X = pd.concat([dummy_df, X], axis=1)
            return X
        
    class CustomLabelEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, object_cols):
            """Encode the data based on object columns provided using label encoding
            """
            self.object_cols = object_cols
            self.encoders = {}

        def fit(self, X, y=None):
            for col in self.object_cols:
                encoder = LabelEncoder()
                encoder.fit(X[col])
                self.encoders[col] = encoder
            return self

        def transform(self, X, y=None):
            for col, encoder in self.encoders.items():
                X[col] = encoder.transform(X[col])
            return X
        
        def inverse_transform(self, X):
            for col, encoder in self.encoders.items():
                X[col] = encoder.inverse_transform(X[col])
            return X
        
    class CustomNumericalEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, object_col):
            """Encode the data based on object column provided
            """
            self.object_col = object_col

        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            sc = StandardScaler()
            X[self.object_col] = sc.fit_transform(X[self.object_col])
            return X        

    class CustomTimeEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, time_col):
            self.object_col = time_col

        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            if self.object_col == 'Horaire théorique':
                X['sin_time'] = np.sin(2 * np.pi * pd.to_timedelta(X[self.object_col].astype(str)).dt.total_seconds()/86400)
                X['cos_time'] = np.cos(2 * np.pi * pd.to_timedelta(X[self.object_col].astype(str)).dt.total_seconds()/86400)
                X = X.drop(self.object_col, axis=1)
            if self.object_col == 'Semaine':
                X['semaine_sin_time'] = np.sin(2 * np.pi * X[self.object_col]/52)
                X['semaine_cos_time'] = np.cos(2 * np.pi * X[self.object_col]/52)
                X = X.drop(self.object_col, axis=1)
            if self.object_col == 'month':
                X['month_sin_time'] = np.sin(2 * np.pi * X[self.object_col]/12)
                X['month_cos_time'] = np.cos(2 * np.pi * X[self.object_col]/12)
                X = X.drop(self.object_col, axis=1)
            if self.object_col == 'Local Date':
                X = X.drop(self.object_col, axis=1)
            return X
        
    
    class CustomTargetRepartitionEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, target_col):
            self.object_col = target_col
        
        def fit(self, y):
            return self
        
        def transform(self, y):
            y[self.object_col] = y[self.object_col].astype(int)
            y['tot_sum'] = y.loc[:, self.object_col].sum(axis=1)
            for c in self.object_col:
                y[c] /= y['tot_sum']
            
            y.drop('tot_sum', axis=1,inplace=True)
            return y
        

    st.header("Prévision des courbes de présentations :")


    with st.status('Données traitées', expanded=True) as status:
        # Display data info
        st.header('Information sur les données')

        @st.cache_data
        def data_processing():
            

            col_test_pred = ['A/D', 'Cie Ope', 'Num Vol', 'Prov Dest', 'Libellé terminal', 'Jour de la semaine', 'Horaire théorique', 
                            'Semaine', 'month', 'Local Date', 'Faisceau géographique', 'Porteur',
                            'Zone A', 'Zone B', 'Zone C', 'Zone A vacance', 'Zone B vacance', 'Zone C vacance', 
                            'is_ferie', 'nom_jour_ferie'] # + [target_col]


            # COLONNES DE PROGRAMME CONCAT
            label_col = ['A/D', 'Cie Ope', 'Num Vol', 'Prov Dest', 'Libellé terminal', 'Jour de la semaine', 'Horaire théorique', 
                            'Semaine', 'month', 'Faisceau géographique', 'Porteur',
                            'Zone A', 'Zone B', 'Zone C', 'Zone A vacance', 'Zone B vacance', 'Zone C vacance', 
                            'is_ferie', 'nom_jour_ferie'] # local date est absent c'est normal
            
            
            # st.write("Colonnes utilisées pour la prédictions: ", col_test_pred, expended=False)
            data_col_pred = deepcopy(data_x[col_test_pred]) # on garde uniquement les colonnes nécessaires pour ce modele de prédiction 
                                                    # (peut etre modifié mais attention à la cohérence avec le modèle qui sera utilisé)
            preprocessing_pipe = Pipeline(
                    steps=[
                        ('label_encoder', CustomLabelEncoder(label_col)),
                        ('time_encoder_horaire', CustomTimeEncoder('Horaire théorique')),
                        ('time_encoder_semaine', CustomTimeEncoder('Semaine')),
                        ('time_encoder_month', CustomTimeEncoder('month')),
                    ]
                )

            data_col_pred = preprocessing_pipe.fit_transform(data_col_pred)
        
            return data_col_pred, preprocessing_pipe
        

        col_pred_repartition = ['ABCDT1', 'EK', 'EL', 'EM', 'F', 'G']
        col_pred_presentation = list(range(0, 26, 1))
        # col_pred_correspondance = ['Taux Correspondance']
        
        data_col_pred, preprocessing_pipe_taux_presentation = data_processing()

        @st.cache_data
        def inverse_classes(series, column):
            series_inversed = deepcopy(series)
            classes = preprocessing_pipe_taux_presentation.named_steps['label_encoder'].encoders[column].classes_
            label_map = {i: label for i, label in enumerate(classes)}
            series_inversed = series_inversed.map(label_map)
            return series_inversed

        data = deepcopy(data_col_pred) # on copie car on inversera les classes pour l'affichage des résultats
    
        min_date = pd.to_datetime(data['Local Date'], format="%d/%m/%Y").min()
        min_date = min_date.strftime("%d/%m/%Y")

        max_date = pd.to_datetime(data['Local Date'], format="%d/%m/%Y").max()
        max_date = max_date.strftime("%d/%m/%Y")

        col = st.columns(4)
        col[0].metric(label="Nb de companies", value=len(data['Cie Ope'].unique()), delta="")
        col[1].metric(label="Nb de vols", value=len(data['Num Vol'].unique()), delta="")
        col[2].metric(label="Date de début SARIAP", value=min_date, delta="")
        col[3].metric(label="Date de fin SARIAP", value=max_date, delta="")

        st.header("Données transformées")
        st.dataframe(data, height=210, use_container_width=True)
        my_bar.progress(10)
        # Zip dataset files
        data.to_csv('pgrm_concat_ml.csv', index=False)
        
        list_files = ['pgrm_concat_ml.csv']
        # list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']

        st.header('Données traitées')
        
        for c in data.columns:
            if c == 'Local Date':
                continue
            if c == 'sin_time':
                continue
            if c == 'cos_time':
                continue
            if c == 'semaine_sin_time':
                continue
            if c == 'semaine_cos_time':
                continue
            if c == 'month_sin_time':
                continue
            if c == 'month_cos_time':
                continue
            
            data_col_pred[c] = inverse_classes(data_col_pred[c], c)
        st.dataframe(data_col_pred, height=210, use_container_width=True)


        with zipfile.ZipFile('pgrm_concat_ml.zip', 'w') as zipF:
            for file in list_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

        with open('pgrm_concat_ml.zip', 'rb') as datazip:
            btn = st.download_button(
                    label='Download ZIP :bar_chart:',
                    data=datazip,
                    file_name="pgrm_concat_ml.zip",
                    mime="application/octet-stream"
                    )
            
    status.update(label="Données traitées", state="complete", expanded=False)
    
    
    
    with st.status("Chargement du modèle ...", expanded=True) as status:
        
        
        
        
        # st.write(xgb.plot_importance(multioutputregressor.estimator_[0], importance_type='weight'), max_num_features=10)
        # st.write('Modèle chargé, colonnes utilisées pour la prédiction: ', data.columns)
    
        my_bar.progress(20)

        @st.cache_data
        def make_predictions(X_test, is_corrected=True):
        
            arrays = [np.array(X_test['Local Date'].values), 
                      np.array(X_test['Cie Ope'].values),
                      np.array(X_test['Num Vol'].values)
                      ] # crééer un tuple de 3 arrays pour l'indexation
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["Local Date", "Cie Ope", "Num Vol"])

            y_pred_presentation = multioutputregressor_presentation.predict(X_test.drop(columns=['Local Date'], axis=1)) # toutes les prédictions sont faites 
            y_pred_presentation = pd.DataFrame(y_pred_presentation, columns=col_pred_presentation, index=index)

            y_pred_repartition = multioutputregressor_repartition.predict(X_test.drop(columns=['Local Date'], axis=1)) # toutes les prédictions sont faites 
            y_pred_repartition = pd.DataFrame(y_pred_repartition, columns=col_pred_repartition, index=index)
            y_pred_repartition = np.maximum(y_pred_repartition, 0)
            y_pred_repartition = np.minimum(y_pred_repartition, 1)

            if is_corrected: # les prédictions ne valent pas souvent 1 au total
                y_pred_presentation = y_pred_presentation.apply(lambda x: x/sum(x), axis=1)
                y_pred_repartition = y_pred_repartition.apply(lambda x: x/sum(x), axis=1)

                return y_pred_presentation, y_pred_repartition
            else:
                return y_pred_presentation, y_pred_repartition 

        y_pred_presentation, y_pred_repartition = make_predictions(data, is_corrected=True)

        st.write('Prédictions effectuées')
        st.dataframe(y_pred_presentation, height=210, use_container_width=True)
        st.dataframe(y_pred_repartition, height=210, use_container_width=True)


    status.update(label="Prédictions effectuées", state="complete", expanded=False)
    my_bar.progress(30)

    with st.status("Courbes de présentation par vols ...", expanded=True) as status:

        @st.cache_data
        def prediction_process(df_presentation, df_repartition):
            prevision_presentation = deepcopy(df_presentation)
            prevision_repartition = deepcopy(df_repartition)

            @st.cache_data
            def prevision_process_internal(prevision):
                prevision = prevision.reset_index()
                prevision['Local Date'] = pd.to_datetime(prevision['Local Date'], format="%d/%m/%Y").dt.date
                prevision['Cie Ope'] = inverse_classes(prevision['Cie Ope'], 'Cie Ope')

                prevision.dropna(subset=['Cie Ope'], inplace=True)

                prevision['Num Vol'] = inverse_classes(prevision['Num Vol'], 'Num Vol')
                prevision['Num Vol'] = prevision['Cie Ope'] + prevision['Num Vol']
                return prevision
        
            prevision_presentation = prevision_process_internal(prevision_presentation)
            prevision_repartition = prevision_process_internal(prevision_repartition)

            @st.cache_data
            def liste_df_unique(df, col):
                return df[col].unique().tolist()
            
            liste_cies = liste_df_unique(prevision_presentation, 'Cie Ope')
            liste_num_vols = liste_df_unique(prevision_presentation, 'Num Vol')

            return {"presentation":prevision_presentation, "repartition":prevision_repartition}, liste_cies, liste_num_vols


        st.header('Téléchargement des courbes de présentation')
        previsions, liste_cies, liste_num_vols = prediction_process(y_pred_presentation, y_pred_repartition)
        st.write('Présentation:')
        st.dataframe(previsions["presentation"], height=210, use_container_width=True)
        st.write('Répartition:')
        st.dataframe(previsions['repartition'], height=210, use_container_width=True)

        # Zip dataset files
        previsions["presentation"].to_csv('courbe_pres_pred.csv', index=False, sep=';', encoding='utf-8')
        previsions["repartition"].to_csv('courbe_rep_pred.csv', index=False, sep=';', encoding='utf-8')

        list_files = ['courbe_pres_pred.csv', 'courbe_rep_pred.csv']

        with zipfile.ZipFile('prediction_pres_rep.zip', 'w') as zipF:
            for file in list_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

        with open('prediction_pres_rep.zip', 'rb') as datazip:
            btn = st.download_button(
                    label='Download ZIP :bar_chart:',
                    data=datazip,
                    file_name="prediction_pres_rep.zip",
                    mime="application/octet-stream"
                    )
    status.update(label="Courbes de présentation par vols", state="complete", expanded=False)

    my_bar.progress(40)

    with st.status("Visualisation des données ...", expanded=True) as status:
       
        @st.cache_data
        def plot_prediction_line(df, col, pred_to_plot, col_color=None):
            plot = alt.Chart(df).mark_bar().encode(
                                    alt.X('Step', title='Step'),
                                    y=f"{col}:Q",
                                    color=col_color,
                                    ).properties(
                                        title=f"Taux de {pred_to_plot}",
                                        width=800,
                                        height=400).configure_axis(
                                                                titleFontSize=15, 
                                                                labelFontSize=12, 
                                                                titleFontWeight='bold',
                                                                labelFontWeight='bold', 
                                                                titleFontStyle='italic', 
                                                                labelFontStyle='italic')
            
            plot.interactive(
                bind_y=True
            )
            st.altair_chart(plot, theme='streamlit', use_container_width=True)
            # st.data_editor(df_prediction, height=400, use_container_width=True)

        @st.cache_data
        def plot_prediction_hist(df, col, col_value, pred_to_plot, col_color=None):
            plot = alt.Chart(df).mark_bar().encode(
                                                alt.X(pred_to_plot, bin=True, title=f'Taux de {pred_to_plot}'),
                                                alt.Y('count():Q', title='Nombre de vols'),
                                                color=col_color,
                                                ).properties(
                                                    title=f"Histogramme de {col} pour {col_value}",
                                                    width=800,
                                                    height=400).configure_axis(
                                                                            titleFontSize=15, 
                                                                            labelFontSize=12, 
                                                                            titleFontWeight='bold',
                                                                            labelFontWeight='bold', 
                                                                            titleFontStyle='italic', 
                                                                            labelFontStyle='italic')
            plot.interactive()
            st.altair_chart(plot, theme='streamlit', use_container_width=True)
       

        @st.cache_data
        def filter_by_date_col(df, date_range, col, col_value):
            return df.loc[(df['Local Date'] >= date_range[0]) 
                          & (df['Local Date'] <= date_range[1])
                          & (df[col].isin(list(col_value)))
                          ].sort_values(by='Local Date').reset_index(drop=True)
        
        @st.cache_data
        def filter_single_date_col(df, date_col, date_value, col, col_value):
            return df.loc[(df[date_col] == date_value) 
                          & (df[col].isin(list(col_value)))
                          ].sort_values(by=['Local Date', 'Step', col]).reset_index(drop=True)


        # INTERFACE
        st.header("Visualisation des prédictions")
        cie_to_plot = selectbox('Choisir une compagnie', liste_cies)
        pred_to_plot = selectbox('Choisir une prédiction', ['presentation', 'repartition'])
        
        if cie_to_plot:
            # num_to_plot = selectbox('Choisir un vol', prevision.loc[prevision['Cie Ope'].str.contains(cie_to_plot)]['Num Vol'].unique().tolist())
            num_to_plot = st.multiselect('Choisir un vol', previsions[pred_to_plot].loc[previsions[pred_to_plot]['Cie Ope'].str.contains(cie_to_plot)]['Num Vol'].unique().tolist())
            # num_to_plot = st.selectbox('Choisir un vol', prevision.loc[prevision['Cie Ope'].str.contains(cie_to_plot)]['Num Vol'].unique().tolist())    
            
            # date_range = date_range_picker('Sélectionner un interval de temps', default_start=pd.to_datetime(min_date, format="%d/%m/%Y"), default_end=pd.to_datetime(max_date, format="%d/%m/%Y"))
            date_input = st.date_input('Sélectionner une date', pd.to_datetime(min_date, format="%d/%m/%Y"), min_value=pd.to_datetime(min_date, format="%d/%m/%Y"), max_value=pd.to_datetime(max_date, format="%d/%m/%Y"))


            prevision_reshaped = previsions[pred_to_plot].melt(id_vars=['Local Date', 'Cie Ope', 'Num Vol'], var_name='Step', value_name=pred_to_plot)
            df_previ_num = filter_single_date_col(df=prevision_reshaped, date_col='Local Date', date_value=date_input, col='Num Vol', col_value=num_to_plot)
            df_previ_cie = filter_single_date_col(df=prevision_reshaped, date_col='Local Date', date_value=date_input, col='Cie Ope', col_value=[cie_to_plot])

            st.header(f"Prédiction de {pred_to_plot}")
            plot_prediction_line(df_previ_num, col=pred_to_plot, pred_to_plot=pred_to_plot, col_color='Num Vol')
            st.header('Prédictions par compagnie')
            plot_prediction_hist(df=df_previ_cie, col='Cie Ope', col_value=cie_to_plot, pred_to_plot=pred_to_plot, col_color='Step')

        
    status.update(label=f"Visualisation des courbes {pred_to_plot}", state="complete", expanded=False)

    my_bar.progress(60)

    st.header("Prévision de l'affluence aux PIFs :")
    
    with st.status("Regroupement données et prédiction ...", expanded=True) as status:
        st.write("Mise en forme des courbes de présentation ...")
        
        @st.cache_data
        def data_pred_merge(data, prevision):
            prevision['Num Vol'] = prevision['Num Vol'].apply(lambda x: x[2:])
            prevision.sort_values(by=['Local Date', 'Cie Ope', 'Num Vol'], inplace=True)
            prevision['Local Date'] = prevision['Local Date'].apply(lambda x: x.strftime("%d/%m/%Y"))
            data.sort_values(by=['Local Date', 'Cie Ope', 'Num Vol'], inplace=True)

            # st.write("Jointure des données avec les prédictions ...")
            df_merged_full = pd.merge(data, prevision, on=['Local Date', 'Cie Ope', 'Num Vol'], how='left')

            # st.header("NaN")
            # st.write(df_merged_full.loc[df_merged_full.isna().any(axis=1)])
            # st.write("Nombre de NaN avant traitement: ", df_merged_full.isna().sum().sum())
            # st.write("Drop des lignes NaN ...")
            df_merged_full.dropna(subset=['Cie Ope'], inplace=True)
            # st.write("Nombre de NaN après traitement: ", df_merged_full.isna().sum().sum())

            # st.write("Données jointes :")
            # st.dataframe(df_merged_full, height=300, use_container_width=True)

            # df_merged_full = pd.read_csv(r"2024-04-15T11-42_export.csv", sep=',', encoding='utf-8').sort_values(by=['Local Date', 'Cie Ope', 'Num Vol']).reset_index(drop=True)
            # df_merged_full.drop(columns=['Unnamed: 0'], inplace=True)

            st.write("Mise en forme des taux de présentation aux différents PIFs ...")

            df_merged_full['Local Date'] = pd.to_datetime(df_merged_full['Local Date'], format="%d/%m/%Y")
            df_merged_full['Local Date'] = df_merged_full['Local Date'].apply(lambda x: x.strftime('%d/%m/%Y'))
            df_merged_full['Horaire théorique'] = pd.to_datetime(df_merged_full['Horaire théorique'], format="%H:%M:%S").dt.time
            df_merged_full[['Pax LOC TOT', 'Pax CNT TOT', 'PAX TOT']] = df_merged_full[['Pax LOC TOT', 'Pax CNT TOT', 'PAX TOT']].astype(int)
            df_merged_full[col_pred_presentation] = df_merged_full[[i for i in range(0, 26, 1)]].astype(float)

            return df_merged_full
        
        df_merged_full_presentation = data_pred_merge(data_x, previsions['presentation'])
        # st.write("Données jointes présentation :")
        # st.dataframe(df_merged_full_presentation, height=300, use_container_width=True)

        df_merged_full_repartition = data_pred_merge(df_merged_full_presentation, previsions['repartition'])

        df_merged_full = deepcopy(df_merged_full_repartition)
        df_merged_full['Datetime'] = pd.to_datetime(df_merged_full['Local Date'] + ' ' + df_merged_full['Horaire théorique'].astype(str), format='%d/%m/%Y %H:%M:%S')
        df_merged_full['Datetime_10min_step'] = df_merged_full['Datetime'].dt.floor('10min')

        st.write("Df merged full :")
        st.dataframe(df_merged_full, height=300, use_container_width=True)

        # st.dataframe(df_merged_full, height=300, use_container_width=True)

        
        # def get_pif():
        #     df = pd.read_excel(r"ressources\fichier_config_PIF.xlsx", sheet_name="Config_ML")
        #     df = df.fillna("XXXXX")
        #     return df[['PIF', 'PIF_pax_type', 'PIF_terminal', 'PIF_sens']]
        
        # df_pif = get_pif()

        # st.write("Données PIF :")
        # st.dataframe(df_pif, height=300, use_container_width=True)


        # SPLIT DU DATAFRAME EN DEUX DATAFRAMES: A (ARRIVEE) ET D (DEPART) 
        # PERMET DE MULTIPLIER LES TAUX PAR LE NOMBRE DE PASSAGERS EN FONCTION DU SENS DE CIRCULATION
        df_merged_d = df_merged_full.loc[df_merged_full['A/D'] == 'D']
        df_merged_d.drop(columns=col_pred_repartition, inplace=True) # SUPPRESSION DES COLONNES COURBE DE REPARTITION CORRESP POUR LES DEPARTS (D)

        df_merged_a = df_merged_full.loc[df_merged_full['A/D'] == 'A']
        df_merged_a.drop(columns=col_pred_presentation, inplace=True) # SUPPRESSION DES COLONNES COURBE DE PRESENTATION POUR LES ARRIVEES (A)

        df_merged_a['PAX Theorique'] = df_merged_a['Pax CNT TOT']

        # LES PASSAGERS DEBARQUENT PAR VAGUE DE 10 MINUTES SELON LES TAUX DE DEBARQUEMENT SUIVANTS:
        debarquement = [0.5, 0.3, 0.2] # 50% DEBARQUENT A LA 1ERE VAGUE, 30% A LA 2EME ET 20% A LA 3EME 
        # on pourrait avoir plusieurs courbe de debarquement en fonction du debarquement des passagers

        deb_1 = deepcopy(df_merged_a)
        deb_1['Datetime_10min_step'] = deb_1['Datetime_10min_step'] + pd.Timedelta(minutes=10)
        deb_1['PAX Theorique'] = deb_1['PAX Theorique'] * debarquement[0]
        deb_1.reset_index(drop=True, inplace=True)
        # dans deb_1 on a tous les passagers qui debarquent à la 1ere vague de 10 minutes


        deb_2 = deepcopy(df_merged_a)
        deb_2['Datetime_10min_step'] = deb_2['Datetime_10min_step'] + pd.Timedelta(minutes=20)
        deb_2['PAX Theorique'] = deb_2['PAX Theorique'] * debarquement[1]
        deb_2.reset_index(drop=True, inplace=True)
        # dans deb_2 on a tous les passagers qui debarquent à la 2eme vague de 10 minutes

        deb_3 = deepcopy(df_merged_a)
        deb_3['Datetime_10min_step'] = deb_3['Datetime_10min_step'] + pd.Timedelta(minutes=30)
        deb_3['PAX Theorique'] = deb_3['PAX Theorique'] * debarquement[2]
        deb_3.reset_index(drop=True, inplace=True)
        # dans deb_3 on a tous les passagers qui debarquent à la 3eme vague de 10 minutes

        df_merged_a = pd.concat([deb_1, deb_2, deb_3], axis=0)

        df_merged_a['PIF'] = df_merged_a['Libellé terminal'] # INITIALISATION DE LA COLONNE PIF QUI EST LA SALLE EMPORT


        df_abcdt1 = df_merged_a[~df_merged_a['Libellé terminal'].isin(['EK', 'EL', 'EM', 'F', 'G', 'Terminal 3'])] # CES PAX LA SORTENT DU TRAJET PIF DONC PLUS BESOIN D'EUX
        df_klmfg = df_merged_a[df_merged_a['Libellé terminal'].isin(['EK', 'EL', 'EM', 'F', 'G'])] # CEUX LA RESTENT DANS LE TRAJET PIF

        
        dict_cnt_a = {
                "EK":{"EK":{"K CTR":10},
                        "EL":{"L CTR":20},
                        "EM":{"M CTR":30},
                        "F":{"Galerie EF":10},
                        "G":{"Galerie EF":10},
                        "ABCDT1":{"K CNT":10}},

                "EL":{"EK":{"K CTR":20},
                        "EL":{"L CNT":10},
                        "EM":{"M CTR":20},
                        "F":{"L CNT":10},
                        "G":{"L CNT":10},
                        "ABCDT1":{"K CNT":10}},

                "EM":{"EK":{"K CTR":30},
                        "EL":{"L CTR":30},
                        "EM":{"M CTR":40},
                        "F":{"Galerie EF":20},
                        "G":{"Galerie EF":40},
                        "ABCDT1":{"K CNT":60}},

                # 'F':{"EK":{"Galerie EF":30}, # IFU
                #         "EL":{"Galerie EF":40}, # IFU
                #         "EM":{"Galerie EF":40}}, # IFU
                
                # 'G':{"EK":{"Galerie EF":30}, # IFU
                #         "EL":{"Galerie EF":40}, # IFU
                #         "EM":{"Galerie EF":40}} # IFU

        }   

        L = []
        for _, row in stqdm(df_klmfg.iterrows()):
            for c in col_pred_repartition:
                new_row = row.copy()
                new_row['PIF'] = c
                new_row['PAX Theorique'] = row['PAX Theorique'] * new_row[c]
                # new_row['Datetime_10min_step'] += pd.Timedelta(minutes=transfer_time)
                # df_klmfg = df_klmfg.append(new_row, ignore_index=True)
                L.append(new_row)

        df_klmfg = pd.DataFrame(L, columns=df_klmfg.columns)

        my_bar.progress(70)

        L = []
        for _, row in stqdm(df_klmfg.iterrows()):
            salle_apport = row['Libellé terminal']
            salle_emport = row['PIF']
            try:
                pif_emport_dict = dict_cnt_a[salle_apport]
            except KeyError:
                if salle_apport == "ABCDT1":
                    L.append(row)
                continue
            # for salle_emport, pif_values in pif_emport_dict.items():
            try:
                for pif_name, transfer_time in pif_emport_dict[salle_emport].items():
                    new_row = row.copy()
                    new_row['PIF'] = pif_name
                    new_row['Datetime_10min_step'] += pd.Timedelta(minutes=transfer_time)
                    # df_klmfg = df_klmfg.append(new_row, ignore_index=True)
                    L.append(new_row)
            except KeyError:
                # print(deboarding_terminal)
                continue

        df_klmfg = pd.DataFrame(L, columns=df_klmfg.columns)

        # st.write('df_merged_a')
        # st.dataframe(df_merged_a, height=300, use_container_width=True)

        # ON EXPLODE LE DATAFRAME POUR OBTENIR UN CRÉNEAU DE 10 MINUTES POUR CHAQUE LIGNE POUR LE SOUS DATAFRAME D'ARRIVE
        df_exploded_a = pd.DataFrame({
            'Datetime': df_klmfg['Datetime_10min_step'], 
            'Local Date': df_klmfg['Local Date'],
            'Cie Ope': df_klmfg['Cie Ope'],
            'Num Vol': df_klmfg['Num Vol'],
            'Libellé terminal': df_klmfg['PIF'],
            'Charge': df_klmfg['PAX Theorique'].values.tolist()
        })

    

        # Explode la colonne 'Datetime' et 'Charge'
        df_exploded_a = df_exploded_a.explode(['Charge'])
        df_exploded_a = df_exploded_a.drop(columns=['Cie Ope', 'Num Vol']).groupby(['Local Date', 'Datetime', 'Libellé terminal']).sum().reset_index()
        df_exploded_a['Local Date'] = df_exploded_a['Datetime'].dt.strftime('%d/%m/%Y')
        

        # DEPART (D)
        for c in col_pred_presentation:
            # Ici on multiplie les taux de présentation par le nombre de passagers pour obtenir le nombre de passagers par PIF
            # Les conditions de filtrage sont les suivantes: 
            # - Conditions de sens de circulation (A/D)
            # - Conditions de terminal (EL, EK, EM, F, G) => (EST)
            # - Conditions de terminal (!EL, !EK, !EM, !F, !G) => (OUEST)
            # - Passagers CNT et TOT pour les vols arrivant (A) et passagers LOC pour les vols partant (D)

            # SI DIRECTION EST D et TERMINAL (EST) ALORS MULTIPLIER PAR PAX    LOC TOT
            # SI DIRECTION OUEST D et TERMINAL (OUEST) ALORS MULTIPLIER PAR    PAX TOT

            df_merged_d.loc[df_merged_full['Libellé terminal'].isin(['EL','EK', 'EM', 'F', 'G']), c] = df_merged_d[c] * df_merged_d['Pax LOC TOT']
            
            # Pour le moment on a uniquement cette ligne car on fait l'approximation que Pax CNT ~= 0 et donc Pax LOC ~= PAX TOT (relation CNT + LOC = TOT)
            df_merged_d.loc[~(df_merged_d['Libellé terminal'].isin(['EL','EK', 'EM', 'F', 'G'])), c] = df_merged_d[c] * df_merged_d['PAX TOT']

        my_bar.progress(80)


        # ON EXPLODE LE DATAFRAME POUR OBTENIR UN CRÉNEAU DE 10 MINUTES POUR CHAQUE LIGNE POUR LE SOUS DATAFRAME DE DEPART
        df_exploded_d = pd.DataFrame({
            'Datetime': df_merged_d.apply(lambda row: pd.date_range(start=row['Datetime_10min_step'] - pd.Timedelta(hours=4), periods=26, freq='10min'), axis=1),
            'Local Date': df_merged_d['Local Date'],
            'Cie Ope': df_merged_d['Cie Ope'],
            'Num Vol': df_merged_d['Num Vol'],
            'Libellé terminal': df_merged_d['Libellé terminal'],
            'Charge': df_merged_d[col_pred_presentation].values.tolist()
        })

        # st.write('Exploded D before explode')
        # st.write(df_exploded_d)

        # Explode la colonne 'Datetime' et 'Charge'
        df_exploded_d = df_exploded_d.explode(['Datetime', 'Charge'])
        df_exploded_d = df_exploded_d.drop(columns=['Cie Ope', 'Num Vol']).groupby(['Local Date', 'Datetime', 'Libellé terminal']).sum().reset_index()
        df_exploded_d['Local Date'] = df_exploded_d['Datetime'].dt.strftime('%d/%m/%Y')



        # pifs_lookup_dict_d = {
        #     'EK': 'K CTR',
        #     'EL': 'L CTR',
        #     'EM': 'M CTR',
        #     'F': 'C2F',
        #     'G': 'C2G',
        #     'Terminal 1': 'Terminal 1',
        #     'Terminal 1_5': 'Terminal 1_5',
        #     'Terminal 1_6': 'Terminal 1_6',
        #     'Terminal 2A': 'Liaison AC',
        #     'Terminal 2B': 'Liaison BD',
        #     'Terminal 2C': 'Liaison AC',
        #     'Terminal 2D': 'Liaison BD',
        #     'Terminal 3': 'T3'
        # }


        # Pour les départ, les pifs d'affectations sont les suivants: 
        pifs_lookup_dict_d = { 
            'EK': 'K CTR',
            'EL': 'L CTR',
            'EM': 'M CTR',
            'F': 'C2F',
            'G': 'C2G',
            'Terminal 1': 'Terminal 1',
            'Terminal 1_5': 'Terminal 1_5',
            'Terminal 1_6': 'Terminal 1_6',
            'Terminal 2A': 'Liaison AC',
            'Terminal 2B': 'Liaison BD',
            'Terminal 2C': 'Liaison AC',
            'Terminal 2D': 'Liaison BD',
            'Terminal 3': 'T3'
        }

        df_exploded_d['Libellé terminal'] = df_exploded_d['Libellé terminal'].map(pifs_lookup_dict_d)
        # list_terminaux = ['Terminal 2A', 'Terminal 2B', 'Terminal 2C', 'Terminal 2D',
        #                 'EK', 'EL', 'EM', 'F', 'G', 'Terminal 3','Terminal 1',
        #                 'Terminal 1_5','Terminal 1_6']
    
        
        st.write('Départ')
        st.dataframe(df_exploded_d, height=300, use_container_width=True)
        st.write('Arrivée')
        st.dataframe(df_exploded_a, height=300, use_container_width=True)


        # CONCATENATION DES DATAFRAMES D'ARRIVEE ET DE DEPART POUR OBTENIR LE DATAFRAME INITIAL        
        st.write("Données prêtes pour la visualisation :")
        
        df_exploded = pd.concat([df_exploded_d, df_exploded_a], axis=0).groupby(['Local Date', 'Datetime', 'Libellé terminal']).sum().reset_index()
        st.dataframe(df_exploded, height=300, use_container_width=True)

        def create_time_vector(date_input, periods=144, freq="10min"):
            """
            Crée un vecteur de dates/heures avec un pas de 10 minutes à partir d'une date en entrée.
            
            Paramètres:
            date_input (pd.Timestamp) - La date de départ.
            
            Retourne:
            pd.Series - Un vecteur de dates/heures avec un pas de 10 minutes.
            """
            start_time = pd.Timestamp(date_input.year, date_input.month, date_input.day, 0, 0, 0)
            time_vector = pd.date_range(start=start_time, periods=periods, freq=freq)
            return pd.Series(time_vector)

        # DataFrame final
        final_df = pd.DataFrame()
        for (local_date, terminal), group in df_exploded.groupby(['Local Date', 'Libellé terminal']):
            # Créer le vecteur de dates/heures pour le groupe
            
            time_vector = create_time_vector(pd.Timestamp(pd.to_datetime(local_date, format="%d/%m/%Y")))
            # Fusionner le vecteur de dates/heures avec le groupe
            temp_df = time_vector.to_frame('Datetime').merge(group, how='left', on='Datetime')
            
            # Remplir les valeurs manquantes
            temp_df['Local Date'] = local_date
            temp_df['Libellé terminal'] = terminal
            temp_df['Charge'] = temp_df['Charge'].fillna(0)

            # SHIFTING THE DATETIME BY 2 HOURS TO MATCH THE REAL DATA
            # temp_df['Datetime'] = temp_df['Datetime'] + pd.Timedelta(hours=4)
            # temp_df = time_vector.to_frame('Datetime').merge(group, how='left', on='Datetime')
            # temp_df = temp_df.bfill().ffill()


            # Ajouter le groupe au DataFrame final
            final_df = pd.concat([final_df, temp_df])

        
        # Afficher le DataFrame final
        final_df.dropna(inplace=True)
        

    status.update(label="Mise en forme de l'export PIF", state="complete", expanded=False)

    my_bar.progress(95)

    with st.status("Export des données ...", expanded=True) as status:

        def UKS(y, smoothing_factor=50):
            ukf = UnscentedKalmanFilter(observation_covariance=smoothing_factor)
            (filtered_state_means, _) = ukf.filter(y)
            return filtered_state_means

        final_df = final_df.groupby(['Local Date', 'Datetime', 'Libellé terminal']).sum().reset_index()
        final_df.sort_values(by=['Local Date', 'Libellé terminal'], inplace=True)

        final_df['Local Date'] = final_df['Local Date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'))
        final_df['Datetime'] = final_df['Datetime'].apply(lambda x: str(x).split(' ')[1])
        final_df.columns = ['jour', 'heure', 'site', 'charge']


        final_df.loc[(final_df['charge'] > 0) & (final_df['site'] == 'M CTR') & (pd.to_datetime(final_df['heure'], format="%H:%M:%S") > pd.to_datetime("16:30:00", format="%H:%M:%S")), 'charge'] = 0
        # final_df.loc[(final_df['charge'] > 0) & (final_df['site'] == 'M CTR') & (pd.to_datetime(final_df['heure'], format="%H:%M:%S") < pd.to_datetime("6:00:00", format="%H:%M:%S")), 'charge'] = 0
        # final_df.loc[(final_df['charge'] > 0) & (final_df['site'] == 'K CNT') & (pd.to_datetime(final_df['heure'], format="%H:%M:%S") < pd.to_datetime("6:00:00", format="%H:%M:%S")), 'charge'] = 0
        # final_df.loc[(final_df['charge'] > 0) & (final_df['site'] == 'L CNT') & (pd.to_datetime(final_df['heure'], format="%H:%M:%S") < pd.to_datetime("6:00:00", format="%H:%M:%S")), 'charge'] = 0
        # final_df.loc[(final_df['charge'] > 0) & (final_df['site'] == 'Galerie EF') & (pd.to_datetime(final_df['heure'], format="%H:%M:%S") < pd.to_datetime("6:30:00", format="%H:%M:%S")), 'charge'] = 0

        for d in stqdm(final_df['jour'].unique()): # DECALAGE DU AU MANQUE DE COURBE POUR BD EN 2022/2023 QUI PROVOQUE MAUVAISES PREDICTIONS
            s = 'Liaison BD'
            final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'] = final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'].shift(-3)
            final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'] = final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'].fillna(0)

        for d in stqdm(final_df['jour'].unique()): #SMOOTHING DES PR2DICTIONS
            for s in final_df['site'].unique():
                final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'] = UKS(final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'].fillna(0).values, 2)
                final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'] = final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'].shift(-2)
                final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'] = final_df.loc[(final_df['jour'] == d) & (final_df['site'] == s), 'charge'].fillna(0)



        st.write("Données prêtes pour la visualisation :")
        st.dataframe(final_df, height=300, use_container_width=True)

        st.write("Export des données ...")


        min_date = pd.to_datetime(final_df['jour'], format="%Y-%m-%d").min()
        min_date = min_date.strftime("%d_%m_%Y")

        max_date = pd.to_datetime(final_df['jour'], format="%Y-%m-%d").max()
        max_date = max_date.strftime("%d_%m_%Y")


        final_df.to_csv(f'pif_previ_ml.csv', index=False, sep=';', encoding='utf-8', decimal=',')
        # final_df_smoothed.to_csv(f'pif_previ_smoothed_ml.csv', index=False, sep=';', encoding='utf-8', decimal=',')
        # list_files = ['pif_previ_ml.csv', 'pif_previ_smoothed_ml.csv']
        final_df.to_excel(f'pif_previ_ml.xlsx', index=False, engine='openpyxl')

        list_files = ['pif_previ_ml.csv', 'pif_previ_ml.xlsx']


        with zipfile.ZipFile(f'pif_previ_ml_{min_date}_to_{max_date}.zip', 'w') as zipF:
            for file in list_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

        with open(f'pif_previ_ml_{min_date}_to_{max_date}.zip', 'rb') as datazip:
            btn = st.download_button(
                    label='Download ZIP :bar_chart:',
                    data=datazip,
                    file_name=f'pif_previ_ml_{min_date}_to_{max_date}.zip',
                    mime="application/octet-stream"
                    )
            download_final_df = True
    


    status.update(label="Données exportées", state="complete", expanded=True)

    my_bar.progress(100)


    
# Ask for CSV upload if none is detected
if uploaded_file_y is None:
    st.warning('👈 Pour commencer, veuillez uploader le fichier datamart.')