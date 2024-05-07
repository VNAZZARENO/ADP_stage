import pandas as pd
import streamlit as st
import os
import time
import pandas as pd
import xlwt
from xlwt.Workbook import *
from pandas import ExcelWriter
import xlsxwriter
import datetime
import calendar
import locale
from openpyxl.styles import Font
import itertools
from datetime import datetime, timedelta
import io
from pyxlsb import open_workbook as open_xlsb
import re
from itertools import product
locale.setlocale(locale.LC_ALL, "fr_FR")

st.title("✅ Macro final")
#add_logo("Logo_Groupe_ADP.png")
st.write("Macro du fichier Export_pif final")

def findDay(date):
    born = datetime.datetime.strptime(date, '%d %m %Y').weekday()
    return (calendar.day_name[born])   


data = []

df_config = pd.DataFrame(data)


##### modification 
#df_config['site'] = ['K CTRCNT', 'L CTR', 'L CNT', 'M CTR', 'Galerie EF', 'C2F', 'C2G', 'Liaison BD',
#                    'T3', 'Terminal 1', 'Terminal 1_5', 'Terminal 1_6']



df_sheet = pd.DataFrame(
    [
       {"Pif(s)": "K_CTR,K_CNT", "Supprimer": False, "Ajouter": True},
       {"Pif(s)": "K_CTR", "Supprimer": True, "Ajouter": False},
       {"Pif(s)": "K_CNT", "Supprimer": True, "Ajouter": False},
   ]
)

st.divider()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine='pyxlsb', header=0)
    # try:
    #     df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', engine='python', decimal=',', header=0)
    # except BadZipFile:
        
    df['jour'] = pd.to_datetime(df['jour'], format='%Y-%m-%d')
    df['jour'] = df['jour'].apply(lambda x: x.strftime("%d/%m/%Y"))
    df['charge'] = df['charge'].astype(float)
    df['site'] = df['site'].astype(str)
    df['heure'] = pd.to_datetime(df['heure'], format='%H:%M:%S').dt.strftime('%H:%M:%S')

    with st.expander("Aperçu des données"):
        st.data_editor(df)
    ### patch
    # création d'un dataframe contenant toutes les combinaisons jour/heure/site
    jours= df['jour'].unique()
    heures = pd.date_range("00:00:00", "23:50:00", freq="10min").strftime('%H:%M:%S')
    sites= df['site'].unique()

    combinaisons = pd.DataFrame(list(product(jours,heures,sites)), columns=['jour', 'heure','site'])
    
    # jointure entre le df_final et le df combinaisons pour corriger le problème d'omission de colonne
    
    df_complet = pd.merge(df, combinaisons, on = ['jour', 'heure','site'], how = "right")
    df_complet['charge'].fillna(0, inplace=True)
    
    df = df_complet
    
    ### fin patch
   
    
   ### regroupement des data PIF K CTR K CNT
    
    
   
    def consolider_pif_k(df):
        if st.checkbox("Fusionner K CTR et K CNT", value =True):
    
    
            df['site'] = df['site'].apply(lambda x: 'K CTRCNT' if x in ['K CNT', 'K CTR'] else x)

            df = df.groupby(['jour','heure','site'], as_index=False)['charge'].sum()
        return df
    
       
       
    df = consolider_pif_k(df)
    
    #df_config['site']= df['site'].unique()
    
    #créer une liste dynamique des sites (revient à faire unique(), mais permet ici de supprmier les sites fermées)
    df_config =df.groupby(['site'], as_index=False)['charge'].sum()
    df_config =df_config[df_config['charge'] > 0]
    df_config = df_config.drop(columns=['charge'])
    
    #création d'un dict. pour trier les onglets selon ordre défini
    custom_dict = {'K CTRCNT' : 0,'K CTR' : 1,'K CNT' : 2 , 'L CTR' : 3, 'L CNT' : 4, 'M CTR' : 5, 'Galerie EF' : 6, 'C2F' : 7, 'C2G' : 8, 'Liaison AC' : 9,'Liaison BD' : 10,
    'T3': 11, 'Terminal 1' : 12, 'Terminal 1_5' : 13, 'Terminal 1_6' : 14}
    
    df_config = df_config.sort_values(by=['site'], key=lambda x: x.map(custom_dict)).reset_index(drop=True)
    
    
    df_config['Abattement (%)'] = 0
    
    #par défaut la valeur d'abattement pour L CNT =20%
    condition = df_config['site'].str.contains('L CNT', case=False) 
    df_config.loc[condition, 'Abattement (%)'] = 20
    ### fin 
    
    
    start_date = pd.to_datetime(uploaded_file.name[14:24])
    end_date = pd.to_datetime(uploaded_file.name[28:38])  

    mask = (df['jour'] >= start_date) & (df['jour'] <= end_date)
    mask_dissocie_1 = (df['jour'] >= start_date) & (df['jour'] <= end_date - timedelta(days=7))
    mask_dissocie_2 = (df['jour'] >= start_date + timedelta(days=4)) & (df['jour'] <= end_date)

    df = df.loc[mask]
    export_pif_4_jours = df.loc[mask_dissocie_1]
    export_pif_4_jours.filename = "export_pif_4jours"
    export_pif_7_jours = df.loc[mask_dissocie_2]
    export_pif_7_jours.filename = "export_pif_7jours"

    st.divider()
 
    col1, col2 = st.columns(2)
    with col1:
        st.write("Gestion de l'abattement et de l'ordre des feuilles :")
        df_config = st.data_editor(df_config)
    #with col2:
     #   st.write("Fusionner K CTR et K CNT :")
      #  df_sheet = st.data_editor(df_sheet,  num_rows="dynamic")


  #  sheet_to_delete = df_sheet[df_sheet['Supprimer']]["Pif(s)"].to_list()
   # sheet_to_sum = df_sheet[df_sheet['Ajouter']]["Pif(s)"].to_list()

  #  sheet_to_sum = [re.split(r"[-;,.\s]\s*", item) for item in sheet_to_sum]
    #st.write(sheet_to_sum)

    
    
    st.divider()

    col11, col22 = st.columns([1, 2])
    on = col11.toggle('Dissocié')

    if not on:
        col22.write('Le fichier est **unique**')
        dataframe = [df]

    if on:
        col22.write('Le fichier sera dissocié en **deux fichiers distinct**')
        dataframe = [export_pif_4_jours, export_pif_7_jours]



    def clean(df,i):
        
        df['Total'] = df.iloc[:, 1:145].sum(axis=1)
        df['Numéro de Jour'] = df['jour'].dt.day
        df['Date complète'] = df['jour'].dt.strftime('%d/%m/%Y')
        df['Jour de la semaine'] = df['jour'].dt.day_name(locale="fr_FR")     
        g = str(i).replace(" ", "_")
        df[str(i).replace(" ", "_")] = df['jour'].dt.month_name(locale="fr_FR")
        df["Jour férié ?"] = ""
        first_column = df.pop('Jour férié ?')
        df.insert(1, '"Jour férié ?', first_column)
        first_column = df.pop('Numéro de Jour')
        df.insert(1, 'Numéro de Jour', first_column)
        first_column = df.pop('Date complète')
        df.insert(3, 'Date complète', first_column)
        first_column = df.pop('Jour de la semaine')
        df.insert(3, 'Jour de la semaine', first_column)
        first_column = df.pop(str(i).replace(" ", "_"))
        df.insert(0, str(i).replace(" ", "_"), first_column)
        df.pop('jour')
        df[str(i).replace(" ", "_")] = list(itertools.chain.from_iterable([key] + [float('nan')]*(len(list(val))-1) 
                            for key, val in itertools.groupby(df[str(i).replace(" ", "_")].tolist())))

    
    def findDay(date):
        born = datetime.datetime.strptime(date, '%d %m %Y').weekday()
        return (calendar.day_name[born])    



    st.divider()
    buffer = io.BytesIO()

    if not on:
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            site = []
            df_site = {}
            for i in df_config.site.unique():
                name = str(i).replace(" ", "_")
                site += [name]
                name = df.copy()
                name = name[name['site'] == i]
                name = name.pivot_table(values='charge', index='jour', columns=['heure'], aggfunc='first')
                name.reset_index(inplace=True)
                name.fillna(0, inplace=True)
                df_site[i] = name
                clean(name,i)
                mask = df_config['site'] == i
                if df_config[mask]['Abattement (%)'].iloc[0] != 0:
                    for o in range(5,len(name.columns)):
                        name.iloc[:,o] *= (100-df_config[mask]['Abattement (%)'].iloc[0])/100
               
                name.to_excel(writer, sheet_name=str(i).replace(" ", "_"), index=False)

            writer.close()

            st.download_button(
            label="Télécharger fichier Export pif",
            data=buffer,
            file_name="export_pif.xlsx",
            mime="application/vnd.ms-excel"
            )

    if on:
        for df in dataframe:
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Write each dataframe to a different worksheet.
                site = []
                for i in df_config.site.unique():
                    name = str(i).replace(" ", "_")
                    site += [name]
                    name = df.copy()
                    name = name[name['site'] == i]
                    name = name.pivot_table(values='charge', index='jour', columns=['heure'], aggfunc='first')
                    name.reset_index(inplace=True)
                    name.fillna(0, inplace=True)
                    clean(name,i)
                    mask = df_config['site'] == i
                    if df_config[mask]['Abattement (%)'].iloc[0] != 0:
                        for o in range(5,len(name.columns)):
                            name.iloc[:,o] *= (100-df_config[mask]['Abattement (%)'].iloc[0])/100
                    name.to_excel(writer, sheet_name=str(i).replace(" ", "_"), index=False)
                writer.close()

                st.download_button(
                label="Télécharger fichier " + df.filename,
                data=buffer,
                file_name= df.filename + ".xlsx",
                mime="application/vnd.ms-excel"
                )
