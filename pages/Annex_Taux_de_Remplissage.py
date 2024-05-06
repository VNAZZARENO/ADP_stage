import itertools
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import altair as alt
import time
import zipfile
import datetime
from functools import reduce
from copy import deepcopy
from pykalman import KalmanFilter, UnscentedKalmanFilter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, TimeSeriesSplit
import xgboost as xgb

from stqdm import stqdm


# Page title
st.set_page_config(page_title='CDGD3 Taux Remplissage', 
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
    st.info("Cette application permet d'utiliser un model machine learning pour prédire les courbes de présentation passager sur une période donnée.")

    st.markdown('**Comment utiliser cette application ?**')
    st.markdown('1. **Charger des fichiers CSV** : Chargez un fichier CSV contenant les données à analyser \n->Saria P')
    st.markdown('2. **Paramètres** : Réglez les paramètres du modèle.')
    st.markdown("3. **Exécution** : L'éxécution du modèle se fait automatiquement une fois les fichiers chargés et les paramètres réglés.")
    st.markdown('4. **Résultats** : Consultez les résultats du modèle sur la période.')
    st.markdown('5. **Téléchargement** : Téléchargez les résultats du modèle.')

    st.markdown('Librairies utilisées:')
    st.code('''- Pandas pour la manipulation des données
    - Scikit-learn pour la construction du modèle
    - XGBoost pour la prédiction des taux de remplissage  
    - Altair pour la visualisation des données
    - Streamlit pour la création de l\'application web 
    ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:

    st.header('1. Paramètres de l\'application')
    sequence_length = st.slider('Horizons de prédiction', 7, 28, 14, 1)
    smoothnes_factor = st.slider('Facteur de lissage', 2, 15, 3, 1)
    sleep_time = st.slider('Sleep time', 0, 3, 1)
    compute_offre_matrix = st.checkbox('Calculer la matrice d\'offre par compagnie et sous-type d\'avion', value=True)

    # Load data
    st.header('2.1. Fichier Source')

    st.markdown('**1. Données Taux Remplissage**')
    separator_taux = st.selectbox('Séparateur données datamart (par défaut tabulation)', [';', ',', '\t', '|'], index=2)
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

    uploaded_file_y = st.file_uploader("Fichier requête datamart Remplissage", type=["csv", "txt"])
    
    if uploaded_file_y is not None:
        data_y = pd.read_csv(uploaded_file_y, sep=separator_taux, date_format='%dd/%mm/%YYYY', encoding='utf-8')
        st.write(data_y.head(5))
      
    st.header('2.2. Fichier Companies SARIAP')
    st.markdown('**2. Données Companies**')
    separator_cie = st.selectbox('Séparateur données companies (par défaut ;)', [';', ',', '\t', '|'], index=0)
    uploaded_file_previ_cie = st.file_uploader("Fichier requête SARIAP Companies", type=["csv"])
    if uploaded_file_previ_cie is not None:
        previs_cie = pd.read_csv(uploaded_file_previ_cie, sep=separator_cie, header=0, date_format='%d/%m/%Y')
        previs_cie['Num Vol'] = previs_cie['CieOpe'].astype(str) + previs_cie['NumVol'].astype(str)
        previs_cie.drop(columns=['CieOpe', 'NumVol', 'EscArr', 'EscDep', 'NbPaxCNT'], axis=1, inplace=True)
        st.write(previs_cie.head(5))


# Initiate the model building process
if uploaded_file_y is not None and uploaded_file_previ_cie is not None: 
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
        
    @st.cache_data
    def get_y_df(csv_file, col_id_group, smoothnes_factor, shift_days_pax=sequence_length, sep=','):
        # y_actual_data = pd.read_csv(path_file, sep=sep, date_format='%dd/%mm/%YYYY')
        csv_file.columns = ['Source', 'Local Date', 'Plage', 'A/D', 'Cie Ope', 'Num Vol', 
                            'Immatriculation', 'Sous-type avion', 'Ville', 'Prov Dest', 
                            'Faisceau géographique', 'PAX TOT', 'Offre']

        calendrier_gouv = pd.read_csv(r"ressources\calendar_complete_from_data_gouv.csv", sep=',', date_format='%dd/%mm/%YYYY')
        calendrier_gouv['date'] = calendrier_gouv['date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%d/%m/%Y"))
        calendrier_gouv.fillna('non_ferie', inplace=True)
        df = pd.merge(csv_file, calendrier_gouv, left_on='Local Date', right_on='date', how='left')
        df.drop(columns=['date'], axis=1, inplace=True)

        # df['PAX TOT'] = df['PAX TOT'].apply(lambda x: x.replace(u'\xa0', u'')).astype(int)
        df['Year'] = pd.to_datetime(df['Local Date'], format="%d/%m/%Y").dt.year
        df['month'] = pd.to_datetime(df['Local Date'], format="%d/%m/%Y").dt.month

        
        # df['Semaine'] = pd.to_datetime(df['Local Date'], format="%d/%m/%Y").apply(lambda x: x.isocalendar()[1])
        # df['Jour de la semaine'] = pd.to_datetime(df['Local Date'], format="%d/%m/%Y").apply(lambda x: x.isocalendar()[-1])  
        
        y_actual_data = deepcopy(df)

        try:
            y_actual_data['PAX TOT'].replace(" ", "", regex=True, inplace=True)
            y_actual_data['PAX TOT'] = y_actual_data['PAX TOT'].apply(lambda x: x.replace(u'\xa0', u''))
        except AttributeError:
            pass
        y_actual_data['PAX TOT'] = y_actual_data['PAX TOT'].astype(float)

        # y_actual_data[['PAX TOT', 'Offre']] = y_actual_data[['PAX TOT', 'Offre']].astype(int)
        ind = y_actual_data.loc[(y_actual_data['Immatriculation'].isna()) & (y_actual_data['Source'] == "SARIAE")].index
        y_actual_data.drop(index=ind, inplace=True)
        ind = y_actual_data.loc[y_actual_data['Source'].isna()].index
        y_actual_data.drop(index=ind, inplace=True)
        immat = pd.read_excel(r"ressources\Base immat.xlsx", header=0)

        y_actual_data_merged = pd.merge(y_actual_data, immat, left_on=['Immatriculation'], right_on=['Immatriculation'], how='left').drop(columns=['Immatriculation', 'Avion'], axis=1)
        ind = y_actual_data_merged.loc[(y_actual_data_merged['Source'] == "SARIAE") & (y_actual_data_merged['Offre sièges'].isna())].index
        y_actual_data_merged.drop(index=ind, inplace=True)


        y_actual_data_merged['Taux Remplissage'] = y_actual_data_merged['PAX TOT'] / y_actual_data_merged['Offre sièges']
        y_actual_data_merged['Offre'] = y_actual_data_merged['Offre sièges'] # On remplace l'offre par l'offre sièges car Offre est fausse (valeur = PaxTOT en base...)
        y_actual_data_merged.drop(columns=['Offre sièges'], axis=1, inplace=True) 
        ind = y_actual_data_merged['PAX TOT'].loc[y_actual_data_merged['PAX TOT'] > 600].index
        y_actual_data_merged.drop(index=ind, inplace=True)
        ind = y_actual_data_merged['Taux Remplissage'].loc[y_actual_data_merged['Taux Remplissage'] > 1.2].index
        y_actual_data_merged.drop(index=ind, inplace=True)

        y_actual_data_merged['Semaine'] = pd.to_datetime(y_actual_data_merged['Local Date'], format="%d/%m/%Y").apply(lambda x: x.isocalendar()[1])
        y_actual_data_merged['Jour de la semaine'] = pd.to_datetime(y_actual_data_merged['Local Date'], format="%d/%m/%Y").apply(lambda x: x.isocalendar()[-1])

        list_col = ['PAX TOT', 'Taux Remplissage']


        st.write(f'Shifting data ({sequence_length} days)...')

        for num in stqdm(y_actual_data_merged['Num Vol'].unique()):
            for c in list_col:
                y_actual_data_merged.loc[y_actual_data_merged['Num Vol'] == num, f'{c} SHIFT'] = y_actual_data_merged.loc[y_actual_data_merged['Num Vol'] == num, c].shift(sequence_length)
        for c in list_col:
            y_actual_data_merged.drop(index=y_actual_data_merged.loc[y_actual_data_merged[f'{c} SHIFT'].isna()].index, inplace=True)
        st.write(f"Data shifté de {sequence_length} jours pour la prédiction à horizon {sequence_length} jours")


        
        time.sleep(sleep_time)

        st.write(f'Colonnes pour le groupement catégorique: {col_id_group}')
        time.sleep(sleep_time)
        y_grouped = y_actual_data_merged.loc[(y_actual_data_merged['Source'] == "SARIAE")].groupby(by=col_id_group)['Taux Remplissage']

        y_grouped = y_grouped.mean().reset_index()
        y_actual_data_grouped = y_actual_data_merged.loc[(y_actual_data_merged['Source'] == "SARIAE")].merge(y_grouped, on=col_id_group, how='left', suffixes=('', '_grouped'))
        y_actual_data_grouped['id'] = y_actual_data_grouped[col_id_group[-1]]
        y_actual_data_grouped['id'] = np.add.reduce(y_actual_data_grouped[col_id_group[1:]].astype(str), axis=1)

        st.write("Groupement effectué")
        time.sleep(sleep_time)

        # On ecrase les valeurs de Taux Remplissage par les valeurs groupées
        y_actual_data_grouped['Taux Remplissage'] = y_actual_data_grouped['Taux Remplissage_grouped']
        y_actual_data_grouped.drop(columns=['Taux Remplissage_grouped'], axis=1, inplace=True)
        y_actual_df = pd.concat([y_actual_data_grouped, y_actual_data_merged.loc[(y_actual_data_merged['Source'] == "SARIAP")]], axis=0).reset_index(drop=True)


        col_test_label = ['Jour de la semaine', 'Plage', 'A/D', 'Cie Ope', 'Num Vol', 'Prov Dest', 'Ville',  'Sous-type avion',
                 'Faisceau géographique', 'Zone A', 'Zone B', 'Zone C', 'Zone A vacance', 'Zone B vacance', 'Zone C vacance', 
                 'Year', 'is_ferie', 'nom_jour_ferie', 'id', 'month', 'Semaine', 'PAX TOT SHIFT']
        
  
        labels = col_test_label[:-1]

        preprocessing_pipe_taux_remplissage = Pipeline(
                steps=[
                    ('label_encoder', CustomLabelEncoder(labels)),
                    # ('time_encoder_horaire', CustomTimeEncoder('Horaire théorique')),
                    ('time_encoder_semaine', CustomTimeEncoder('Semaine')),
                    ('time_encoder_month', CustomTimeEncoder('month'))
                    # ('date_droping', CustomTimeEncoder('Local Date')),
                ]
            )

        y_processed_data = preprocessing_pipe_taux_remplissage.fit_transform(y_actual_df)

        classes_y_id = preprocessing_pipe_taux_remplissage.named_steps['label_encoder'].encoders['id'].classes_
        classes_y_num_vol = preprocessing_pipe_taux_remplissage.named_steps['label_encoder'].encoders['Num Vol'].classes_

        label_map_y_num_vol = {i: label for i, label in enumerate(classes_y_num_vol)}
        label_map_y_id = {i: label for i, label in enumerate(classes_y_id)}


        y_processed_data['Num Vol'] = y_processed_data['Num Vol'].map(label_map_y_num_vol)
        y_processed_data['id'] = y_processed_data['id'].map(label_map_y_id)


        return y_processed_data, preprocessing_pipe_taux_remplissage

    with st.status("En cours ...", expanded=True) as status:
    
        st.header("Chargement des données ...")

        y_processed_data, preprocessing_pipe_taux_remplissage = get_y_df(data_y, col_unique_id_group_y, sequence_length, separator_taux)
        time.sleep(sleep_time)

        st.write("Données chargées")
        st.dataframe(y_processed_data, height=210, use_container_width=True)

    status.update(label="Données chargées", state="complete", expanded=False)
        

    def process_round_for_NN(data, col_dict):
        data_copy = deepcopy(data)
        for col, v in col_dict.items():
            data_copy[col] = np.round(data[col]/v)*v
        return data_copy


    def UKS(y, smoothing_factor=50):
        ukf = UnscentedKalmanFilter(observation_covariance=smoothing_factor)
        (filtered_state_means, _) = ukf.filter(y)
        return filtered_state_means


    def process_data_for_NN(df, retained_columns):
        unique_flight_data = deepcopy(df)
        return unique_flight_data[retained_columns]

    def process_normalization_for_NN(df, sequence_length, col_to_normalize):
        unique_flight_data = deepcopy(df)
        for col, d in col_to_normalize.items():
            unique_flight_data[f"{col} NORM"] = ((unique_flight_data[col] - unique_flight_data[col].rolling(window=sequence_length).mean())/unique_flight_data[col].rolling(window=sequence_length).std())
            unique_flight_data[f"{col} STD"] = unique_flight_data[col].rolling(window=sequence_length).std()
            if d:
                unique_flight_data.drop(columns=col, axis=1, inplace=True)
        unique_flight_data.dropna(inplace=True)
        return unique_flight_data


    def process_trigo_for_NN(df, col_to_trigo):
        unique_flight_data = deepcopy(df)
        for col in col_to_trigo:
            unique_flight_data[f"{col} sin"] = np.sin(unique_flight_data[col])
            unique_flight_data[f"{col} cos"] = np.cos(unique_flight_data[col])
            # unique_flight_data.drop(columns=col, axis=1, inplace=True)
        return unique_flight_data


    with st.status('Données traitées', expanded=True) as status:
        # Display data info
        st.header('Information sur les données')

        unique_cie = previs_cie['Num Vol'].unique()
        st.write("Unique Cie")
        st.write(unique_cie.tolist())
    
        filtered_y = deepcopy(y_processed_data)
        filtered_y = y_processed_data.loc[~y_processed_data['Num Vol'].isin(unique_cie) & (~y_processed_data['Num Vol'].str.contains('AF'))] # Take the opposite of unique cie because we predict flights that are not in the previs_cie

        st.write("debug")
        st.dataframe(filtered_y.loc[filtered_y['Source'] == "SARIAP"], height=210, use_container_width=True)

        min_date = pd.to_datetime(filtered_y.loc[filtered_y['Source'] == "SARIAP"]['Local Date'], format='%d/%m/%Y').min()
        min_date = min_date.strftime("%d/%m/%Y")

        max_date = pd.to_datetime(filtered_y.loc[filtered_y['Source'] == "SARIAP"]['Local Date'], format='%d/%m/%Y').max()
        max_date = max_date.strftime("%d/%m/%Y")

        col = st.columns(4)
        col[0].metric(label="Nb de companies", value=len(filtered_y['Cie Ope'].unique()), delta="")
        col[1].metric(label="Nb de vols", value=len(filtered_y['Num Vol'].unique()), delta="")
        col[2].metric(label="Date de début SARIAP", value=min_date, delta="")
        col[3].metric(label="Date de fin SARIAP", value=max_date, delta="")

        st.write("Données traitées:")
        st.dataframe(filtered_y, height=210, use_container_width=True)

        # Zip dataset files
        filtered_y.to_csv('filtered_y.csv', index=False)
        
        list_files = ['filtered_y.csv']
        # list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']

        with zipfile.ZipFile('filtered_y.zip', 'w') as zipF:
            for file in list_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

        with open('filtered_y.zip', 'rb') as datazip:
            btn = st.download_button(
                    label='Download ZIP :bar_chart:',
                    data=datazip,
                    file_name="filtered_y.zip",
                    mime="application/octet-stream"
                    )
            
    status.update(label="Données traitées", state="complete", expanded=False)
    
    with st.status("Matrice d'offre par compagnie et sous-type d'avion", expanded=True) as status:
    
        @st.cache_data
        def matrix_offre_siege(data, cie_list):
            offre_siege = {}
            data_map = deepcopy(data)

            classes_y_cie = preprocessing_pipe_taux_remplissage.named_steps['label_encoder'].encoders['Cie Ope'].classes_

            label_map_cie = {i: label for i, label in enumerate(classes_y_cie)}

            data_map['Cie Ope'] = data_map['Cie Ope'].map(label_map_cie)
            cie_list_map = [label_map_cie[cie] for cie in cie_list]
            for p in stqdm(data_map['Sous-type avion'].unique()):
                offre_siege[p] = {}
                for cie in cie_list_map:
                    try:
                        subset = data_map.loc[(data_map['Num Vol'].str.contains(cie)) & (data_map['Sous-type avion'] == p) & (data_map['Source'] == "SARIAE")]
                        offre_siege[p][cie] = subset['Offre'].astype(float).mean() + subset['Offre'].astype(float).std()
                    except ValueError:
                        offre_siege[p][cie] = 0
                        continue

            l = cie_list

            offre_siege_df = pd.DataFrame(offre_siege).T.sort_index()
            offre_siege_df.reset_index(inplace=True)
            offre_siege_df.columns = ['Sous-type avion'] + l

            offre_siege_df['Sous-type avion'] = offre_siege_df.index
            offre_siege_df[l] = offre_siege_df[l].fillna(0)
            offre_siege_df[l] = offre_siege_df[l].astype(int)

            offre_siege_df.index = offre_siege_df['Sous-type avion']
            offre_siege_df = offre_siege_df.drop(columns=['Sous-type avion'], axis=1)

            for c in offre_siege_df.columns:
                non_zero_rows, zero_rows_index = offre_siege_df.loc[offre_siege_df[c] != 0], offre_siege_df.loc[offre_siege_df[c] == 0].index
                offre_siege_df.loc[zero_rows_index, c] = non_zero_rows[c].mean()
            offre_siege_df.fillna(0, inplace=True)

            offre_siege_df = offre_siege_df.astype(int)

            classes_y_cie = preprocessing_pipe_taux_remplissage.named_steps['label_encoder'].encoders['Cie Ope'].classes_
            label_map_cie = {i: label for i, label in enumerate(classes_y_cie)}
            offre_siege_df.columns = offre_siege_df.columns.map(label_map_cie)

            offre_siege_df.to_csv(r"ressources\offre_siege_analyse.csv", sep=";", decimal=',')

            return offre_siege_df

    
        if compute_offre_matrix:
            st.write("Calcul de la matrice d'offre par compagnie et sous-type d'avion ...")
            offre_siege_df = matrix_offre_siege(data=filtered_y, cie_list=filtered_y['Cie Ope'].unique().tolist())
            st.write("Matrice d'offre par compagnie et sous-type d'avion calculée")
        else:

            st.write("Utilisation de l'ancienne matrice d'offre par compagnie et sous-type d'avion")
            offre_siege_df = pd.read_csv(r"ressources\offre_siege_analyse.csv", sep=";", decimal=',', header=0, index_col=0)

            st.dataframe(offre_siege_df, height=250, use_container_width=True)
    
        @st.cache_data
        def remplacer_mauvaise_offre(data, matrice, _l_vols_mauvaise_offre, label_map_cie):
            data_map = deepcopy(data)
            data_map['Cie Ope'] = data_map['Cie Ope'].map(label_map_cie)
            for cie in stqdm(_l_vols_mauvaise_offre):
                # for p in data_map.loc[data_map['Cie Ope'] == cie]['Sous-type avion'].unique():
                for p in data_map['Sous-type avion'].unique().tolist():
                    try:
                        data_map.loc[(data_map['Cie Ope'] == cie) & (data_map['Sous-type avion'] == p) & (data_map['Source'] == 'SARIAP'), 'Offre'] = matrice[cie][p]
                    except ValueError:
                        print(f"Error for {cie} and {p}")
                        pass
                    except KeyError:
                        print(f"Error for {cie} and {p}")
                        continue

            return data_map

        # l_mauvaise_offre = ['LO', 'CX', 'LX', 'EW', 'SK', 'OS', 'S4', 'UA', 'NH', 'J2', 'CY', 'HY', 'DL', 'VN', 'RO', 'IB', 'OG', 'A9', 'MU']

        st.write("Correction des offres ...")
        
        # offre_siege_df = pd.read_csv(r"ressources\offre_siege_analyse.csv", sep=";", decimal=',', header=0, index_col=0)
        st.dataframe(offre_siege_df.head(), height=250, use_container_width=True)

        time.sleep(sleep_time)
        classes_y_cie = preprocessing_pipe_taux_remplissage.named_steps['label_encoder'].encoders['Cie Ope'].classes_
        label_map_cie = {i: label for i, label in enumerate(classes_y_cie)}

        filtered_y_corrige = remplacer_mauvaise_offre(data=filtered_y, matrice=offre_siege_df, _l_vols_mauvaise_offre=offre_siege_df.columns, label_map_cie=label_map_cie)

        st.write(f"Offres corrigées, Nombre de lignes contenant des NaN : {filtered_y_corrige.loc[filtered_y_corrige['Offre'].isna(), 'Offre'].shape[0]}")

    status.update(label="Matrice d'offre par compagnie et sous-type d'avion", state="complete", expanded=False)


    with st.status("Prédictions en cours ...", expanded=True) as status:
        @st.cache_data
        def get_data_xgboost(data, col_test_pred, col_to_drop, target_col, target_shift, window_ma, window, sequence_length, secondary_target_col, smoothnes_factor, is_target_kalman=False):
            """ 
            Get data for xgboost model with target_col as target variable and secondary_target_col as target variable to predict 
            with a window of window days and a sequence_length of sequence_length days for each flight in data. 

            Inputs:
            data: pd.DataFrame
            col_test_pred: list
            col_to_drop: list
            target_col: str
            target_shift: str
            window: int
            sequence_length: int
            secondary_target_col: str
            is_target_kalman: bool

            Returns:
            pd.DataFrame
            """

            subset = deepcopy(data)
            tr = process_data_for_NN(subset, col_test_pred + [target_col, target_shift])
            if is_target_kalman:
                tr['Taux Remplissage'] = UKS(tr['Taux Remplissage'].values, smoothing_factor=smoothnes_factor)
                tr['Taux Remplissage SHIFT'] = UKS(tr['Taux Remplissage SHIFT'].values, smoothing_factor=smoothnes_factor)

            tr['Delta'] = tr['Taux Remplissage'] - tr['Taux Remplissage SHIFT']
            tr['Delta MA'] = tr['Delta'].rolling(window=window_ma).mean()
            for i in range(window*2, window*2+1):
                tr[f'{secondary_target_col} n-{i}'] = tr[secondary_target_col].shift(i)
                tr[f'{target_shift} DIFF n-{i}'] = tr[target_shift].diff(i).bfill()


            tr = process_normalization_for_NN(tr, sequence_length//2, {
                                                                        'PAX TOT SHIFT':True, 
                                                                        target_shift:False, 
                                                                    f'{secondary_target_col} n-14':False,
                                                                        #    'Jour de la semaine':False,
                                                                    #    'Taux Remplissage SHIFT DIFF':False
                                                                    })
            

            tr = process_round_for_NN(tr, {target_shift:0.01, 
                                            'PAX TOT SHIFT NORM': 0.1,
                                            'Taux Remplissage': 0.01,
                                            #  'Taux Remplissage SHIFT DIFF':0.01,
                                            'Taux Remplissage SHIFT':0.01,
                                            'Taux Remplissage SHIFT NORM': 0.025,
                                            'Delta MA':0.005,
                                            #  'Taux Remplissage SHIFT DIFF NORM':0.1,
                                            #  'Jour de la semaine NORM':0.1
                                            })
            
            
            tr[f'{secondary_target_col} n-14 x {secondary_target_col} n-14 STD'] = np.multiply(20 * UKS(tr[f'{secondary_target_col} n-14 STD'], smoothnes_factor).flatten(), tr[f'{secondary_target_col} n-14'].values)

            tr.drop(columns=col_to_drop, axis=1, inplace=True)

            tr.dropna(inplace=True, axis=0)
            tr.reset_index(drop=True, inplace=True)

            
            # Add X times Y explanatory variables
            col_list = ['Taux Remplissage SHIFT', 'Delta MA n-14', 'Taux Remplissage SHIFT NORM', 'Taux Remplissage SHIFT DIFF n-14', 'Jour de la semaine']
            combs = []

            for i in range(1, len(col_list)+1):
                combs.append(i)
                els = [list(x) for x in itertools.combinations(col_list, 2)]
                combs.append(els)
            
            tr = process_trigo_for_NN(tr, col_list)

                
            tr.dropna(inplace=True, axis=0)
            tr.reset_index(drop=True, inplace=True)

            return tr
    
        def make_predictions(model, X_seq_test, y_seq_test=None):
            pred = model.predict(X_seq_test)
            y = pd.DataFrame(pred, columns=['pred'])
            if y_seq_test is not None:
                y['actual'] = y_seq_test
                y['delta'] = y['actual'] - y['pred']
            return y
        

        @st.cache_data
        def previe_cie_pred(filtered_y, sequence_length, col_test_pred, target_col, target_shift, secondary_target_col, window_ma, window):
            y = {}
            best_xg_reg = xgb.XGBRegressor()
            best_xg_reg.load_model(r"ressources\taux_remplissage_xgboost_model.model")
            st.write("Modèle chargé")
            st.write("Itérations sur les vols pour prédire la target variable...")
            for num in stqdm(filtered_y['Num Vol'].unique()):
                try:
                    ### Prepare the data for training the model ###
                    sariaE = filtered_y.loc[(filtered_y['Num Vol'] == num) & (filtered_y['Source'] == "SARIAE")].tail(56).reset_index(drop=True)
                    # ts = get_data_xgboost(sariaE, 
                    #                     col_test_pred, col_to_drop=col_to_drop, 
                    #                     target_col=target_col, target_shift=target_shift, 
                    #                     window_ma=window_ma, window=window, sequence_length=sequence_length, secondary_target_col=secondary_target_col,
                    #                     is_target_kalman=True)

                    # X_seq_train, y_seq_train = ts.drop(columns=[secondary_target_col], axis=1), ts[[secondary_target_col]]
                
                    
                    # # best_xg_reg.fit(X_seq_train, y_seq_train)

                    ### Make predictions ### 
                    sariaP = filtered_y.loc[(filtered_y['Num Vol'] == num) & (filtered_y['Source'] == "SARIAP")].reset_index(drop=True)
                    subset_test = pd.concat([sariaE, sariaP], axis=0).reset_index(drop=True)
                    # subset_test = sariaP.reset_index(drop=True)
                    
                    subset_test = subset_test.ffill()
                    local_date_test = subset_test['Local Date']
                    # st.write(f"Num Vol: {num} - Nombre de lignes: {subset_test.shape[0]}")

                    ### Prepare the data for predicting the model ###
  
                    te = get_data_xgboost(subset_test, 
                                        col_test_pred, col_to_drop=col_to_drop, 
                                        target_col=target_col, target_shift=target_shift, 
                                        window_ma=window_ma, window=window, sequence_length=sequence_length, secondary_target_col=secondary_target_col,
                                        smoothnes_factor=smoothnes_factor,
                                        is_target_kalman=True)
                    X_seq_test = te.drop(columns=[secondary_target_col], axis=1)
                    # st.write(f"Num Vol: {num} - Nombre de lignes: {X_seq_test.shape[0]}")

                    y[num] = make_predictions(best_xg_reg, X_seq_test)

                    local_date_test = local_date_test[-y[num].shape[0]:].reset_index(drop=True)
                    y[num]['Local Date'] = local_date_test

                except ValueError:
                    # print(f"Error for flight {num}")
                    continue

            st.write("Prédictions faites")
            # print(f"y length: {len(y)}")

            if y == {}:
                st.write("Aucune prédiction n'a été faite, erreur...")
                return None
            else:
                st.write('Création des prédictions pour les vols et la période donnée')
                previe_cie_prediction = {}
                for num in stqdm(y.keys()):
                    try:
                        prediction_num = y[num]
                        sub_num = filtered_y.loc[(filtered_y['Num Vol'] == num) & (filtered_y['Source'] == 'SARIAP') & (filtered_y['Local Date'].isin(prediction_num['Local Date']))].reset_index(drop=True)

                        local_date_sub_num = sub_num['Local Date']
                        prediction_num = prediction_num.loc[prediction_num['Local Date'].isin(local_date_sub_num)]
                        sub_num.loc[:, 'Taux Remplissage'] = prediction_num['pred'].values + sub_num['Taux Remplissage SHIFT'].values # We add the prediction to the Taux Remplissage SHIFT
                        sub_num.loc[:, 'Offre'] = sub_num['Offre']
                        sub_num.loc[sub_num['Taux Remplissage'] < 0.6, 'Taux Remplissage'] = sub_num['Taux Remplissage'].mean() + 0.1 # We set the Taux Remplissage to the mean of the Taux Remplissage if it is below 0.6 

                        if sub_num.loc[:, 'Offre'].any() == 0:
                            offre_to_insert = filtered_y.loc[(filtered_y['Num Vol'] == num) & (filtered_y['Source'] == 'SARIAE')]['Offre'].mean()
                            # print(f"Offre to insert: {offre_to_insert} for flight {num}")
                            sub_num.loc[sub_num['Offre'] == 0, 'Offre'] = offre_to_insert

                        sub_num.loc[:, 'NbPaxTOT'] = sub_num['Taux Remplissage'] * sub_num['Offre'].astype(float) # We multiply the Taux Remplissage by the Offre to get the PAX TOT
                        sub_num.loc[:,'NbPaxTOT'] = sub_num['NbPaxTOT'].astype(int) # We convert the PAX TOT to integer
                        sub_num.loc[:,'NbPaxCNT'] = sub_num['NbPaxTOT'].astype(int) # We copy the PAX TOT to the PAX CNT to create the PAX CNT column
                        sub_num.loc[:,'NbPaxCNT'] = 0 # We set the PAX CNT to 0

                        sub_num.loc[:, 'Offre'] = sub_num['Offre'].astype(int) # We convert the Offre to integer 
                        
                        # sub_num.loc[:, 'ArrDep'] = sub_num.loc[:, 'A/D']
                        # sub_num.loc[:, 'CieOpe'] = sub_num.loc[:, 'Num Vol'].apply(lambda x: x[:2])
                        # sub_num.loc[:, 'NumVol'] = sub_num.loc[:, 'Num Vol'].apply(lambda x: x[2:])
                        # sub_num.loc[:, 'EscDep'] = 0
                        # sub_num.loc[:, 'EscArr'] = 0
                        # sub_num.loc[:, 'DateLocaleMvt'] = sub_num.loc[:, 'Local Date']
                        
                        previe_cie_prediction[num] = sub_num[['A/D', 'Cie Ope', 'Num Vol', 'Prov Dest', 'Local Date', 'NbPaxTOT', 'NbPaxCNT', 'Offre', 'Taux Remplissage']] # We keep only the columns we need
                        # previe_cie_prediction[num] = sub_num[['ArrDep', 'CieOpe', 'NumVol', 'EscDep', 'EscArr', 'DateLocaleMvt', 'NbPaxTOT', 'NbPaxCNT', 'Offre', 'Taux Remplissage']] # We keep only the columns we need
                    
                    except ValueError:
                        st.write(f"Error for flight {num}")
                        continue
                previe_full = pd.concat([v for k,v in previe_cie_prediction.items()], axis=0).reset_index(drop=True)

                st.write("Mise en forme des prédictions terminée")

                classes_prov_dest = preprocessing_pipe_taux_remplissage.named_steps['label_encoder'].encoders['Prov Dest'].classes_
                label_prov_dest = {i: label for i, label in enumerate(classes_prov_dest)}

                previ = deepcopy(previe_full)

                previ['A/D'] = previ['A/D'].apply(lambda x: 'A' if x == 1 else 'D')
                previ['Prov Dest'] = previ['Prov Dest'].map(label_prov_dest)
                previ['Cie Ope'] = previ['Cie Ope'].apply(lambda x: x[:2])
                previ['Num Vol'] = previ['Num Vol'].apply(lambda x: x[2:])

                previ.columns = ['ArrDep', 'CieOpe', 'NumVol', 'EscDep', 'DateLocaleMvt', 'NbPaxTOT', 'NbPaxCNT', 'Offre', 'Taux Remplissage']
                previ['EscArr'] = previ['EscDep']
                previ.loc[(previ['ArrDep'] == 'D'), 'EscDep'] = 'CDG'
                previ.loc[(previ['ArrDep'] == 'A'), 'EscArr'] = 'CDG'

                return previ[['ArrDep', 'CieOpe', 'NumVol', 'EscDep', 'EscArr', 'DateLocaleMvt', 'NbPaxTOT', 'NbPaxCNT', 'Offre', 'Taux Remplissage']]
                



        #################### Make predictions ####################
        target_col = 'Taux Remplissage'
        target_shift = 'Taux Remplissage SHIFT'
        secondary_target_col = 'Delta MA'

        id_col = 'id'
        # col_test_pred = ['Sous-type avion', 'Prov Dest', 'Faisceau géographique', 'semaine_sin_time', 'semaine_cos_time', 'Jour de la semaine', 'Zone A vacance', 'Zone B vacance', 'Zone C vacance', 'is_ferie', 'nom_jour_ferie', 'PAX TOT SHIFT']
        col_test_pred = ['Jour de la semaine', 'Zone A vacance', 'Zone B vacance', 'Zone C vacance', 'is_ferie', 'nom_jour_ferie', 'PAX TOT SHIFT']
        col_to_drop = ['Taux Remplissage', 'Delta', 'PAX TOT SHIFT NORM']

        window = 7
        window_ma = 3
        sequence_length = 14

        st.header("Prédictions des taux de remplissage")

        previ_full = previe_cie_pred(filtered_y_corrige, sequence_length, col_test_pred, target_col, target_shift, secondary_target_col, window_ma, window)    
        
    status.update(label="Prédictions terminées", state="complete", expanded=False)


    with st.status('Prédictions', expanded=True) as status:
        st.write("Prédictions faites")

        
        min_date = pd.to_datetime(previ_full['DateLocaleMvt'], format='%d/%m/%Y').min()
        min_date = min_date.strftime("%Y_%m_%d")

        max_date = pd.to_datetime(previ_full['DateLocaleMvt'], format='%d/%m/%Y').max()
        max_date = max_date.strftime("%Y_%m_%d")

        # previe_full.to_csv(rf"previe_cie_pred_{min_date}_to_{max_date}.csv", sep=";", index=False, decimal=',')
        st.write(f"Prédictions du {min_date} au {max_date} téléchargeables ci-dessous")


        # Zip dataset files
        previ_full.to_csv(f'previ_cie_{min_date}_to_{max_date}.csv', decimal=',', sep=";", index=False)

        list_files = [f'previ_cie_{min_date}_to_{max_date}.csv']

        with zipfile.ZipFile(f'previ_cie_{min_date}_to_{max_date}.zip', 'w') as zipF:
            for file in list_files:
                zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

        with open(f'previ_cie_{min_date}_to_{max_date}.zip', 'rb') as datazip:
            btn = st.download_button(
                    label='Download ZIP :bar_chart:',
                    data=datazip,
                    file_name=f'previ_cie_{min_date}_to_{max_date}.zip',
                    mime="application/octet-stream", 
                    )
            
        st.header('Prédictions')
        st.dataframe(previ_full, height=250, use_container_width=True)

            
    status.update(label="Prédictions téléchargeables", state="complete", expanded=True)

    # Display model parameters
    with st.status('Prédiction sur la période', expanded=True) as status:
        @st.cache_data
        def liste_cie_opes():
            return previ_full['CieOpe'].unique().tolist()
        
        liste_cies = liste_cie_opes()

        # @st.cache_data
        # def liste_num_vols():
        #     return previe_full['Num Vol'].unique().tolist()
        
        # liste_nums = liste_num_vols

        cie_to_plot = st.selectbox('Choisir une compagnie', liste_cies)
        num_to_plot = st.selectbox('Choisir un vol', previ_full.loc[previ_full['CieOpe'].str.contains(cie_to_plot)]['NumVol'].unique().tolist())    
        

        st.header('Prédictions par vol')

        @st.cache_data
        def plot_prediction_num(num):
            df_prediction = previ_full.loc[previ_full['NumVol'] == num].reset_index(drop=True)
            plot_nbpaxtot = alt.Chart(df_prediction).mark_bar().encode(
                                    x='DateLocaleMvt', 
                                    y='NbPaxTOT',
                                    color='Taux Remplissage',
                                    ).properties(
                                        title=f"Prédictions pour le vol {num}",
                                        width=800,
                                        height=400).configure_axis(
                                                                titleFontSize=15, 
                                                                labelFontSize=12, 
                                                                titleFontWeight='bold',
                                                                labelFontWeight='bold', 
                                                                titleFontStyle='italic', 
                                                                labelFontStyle='italic')
            
            plot_nbpaxtot.interactive()
            st.altair_chart(plot_nbpaxtot, theme='streamlit', use_container_width=True)

            plot_hist_taux_remplissage = alt.Chart(df_prediction).mark_bar().encode(
                                                alt.X('Taux Remplissage', bin=True),
                                                y='count()',
                                                color='NbPaxTOT'
            ).properties(
                title=f"Histogramme des taux de remplissage pour le vol {num}",
                width=800,
                height=400).configure_axis(
                                        titleFontSize=15, 
                                        labelFontSize=12, 
                                        titleFontWeight='bold',
                                        labelFontWeight='bold', 
                                        titleFontStyle='italic', 
                                        labelFontStyle='italic')
            plot_hist_taux_remplissage.interactive()
            st.altair_chart(plot_hist_taux_remplissage, theme='streamlit', use_container_width=True)

            st.data_editor(df_prediction, height=400, use_container_width=True)

        plot_prediction_num(num_to_plot)

        st.header('Prédictions par compagnie')

        def plot_prediction_cie(cie):
            df_prediction = previ_full.loc[previ_full['CieOpe'].str.contains(cie)].reset_index(drop=True)
            plot_nbpaxtot = alt.Chart(df_prediction).mark_circle().encode(
                                    alt.X('Offre'),
                                    alt.Y('NbPaxTOT'),
                                    color='Taux Remplissage',
                                    
                                    # color='Taux Remplissage',
                                    )
            
            plot_nbpaxtot.interactive()
            st.altair_chart(plot_nbpaxtot, theme='streamlit', use_container_width=True)

            plot_hist_taux_remplissage = alt.Chart(df_prediction).mark_bar().encode(
                                                alt.X('Taux Remplissage', bin=True),
                                                y='count()',
                                                color='NbPaxTOT'
            ).properties(
                title=f"Histogramme des taux de remplissage pour la compagnie {cie}",
                width=800,
                height=400).configure_axis(
                                        titleFontSize=15, 
                                        labelFontSize=12, 
                                        titleFontWeight='bold',
                                        labelFontWeight='bold', 
                                        titleFontStyle='italic', 
                                        labelFontStyle='italic')
            plot_hist_taux_remplissage.interactive()
            st.altair_chart(plot_hist_taux_remplissage, theme='streamlit', use_container_width=True)

            st.data_editor(df_prediction, height=400, use_container_width=True)

        plot_prediction_cie(cie_to_plot)  
        

        

    
# Ask for CSV upload if none is detected
if uploaded_file_y is None:
    st.warning('👈 Pour commencer, veuillez uploader le fichier datamart.')
elif uploaded_file_y is not None and uploaded_file_previ_cie is None:
    st.warning('👈 Ensuite, veuillez uploader le fichier prévisions compagnies.')
