import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
from itertools import product
import locale

st.set_page_config(page_title="Verif Seuil", page_icon="📏", layout="centered", initial_sidebar_state="auto", menu_items=None)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


st.title('📏 PIF - Charge horaire / seuil de saturation')
    
with st.sidebar:
        st.write("## 📂 Importation des données")
        separator_previ_pif = st.selectbox('Séparateur données datamart (par défaut tabulation)', [';', ',', '\t', '|'], index=0)
        uploaded_file = st.file_uploader("Selectionner un fichier", type=["xls", "xlsx", "csv", "txt"])
        
        if uploaded_file is not None: 
                try:
                        df = pd.read_excel(uploaded_file)
                except ValueError:
                        df = pd.read_csv(uploaded_file, sep=separator_previ_pif)
                        df['jour'] = df['jour'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
                st.data_editor(df)
if uploaded_file is not None:
        # création d'un dataframe contenant toutes les combinaisons jour/heure/site
        jours= df['jour'].unique()
        heures = pd.date_range("00:00:00", "23:50:00", freq="10min").strftime('%H:%M:%S')
        sites= df['site'].unique()
    
        combinaisons = pd.DataFrame(list(product(jours,heures,sites)), columns=['jour', 'heure','site'])
       
# jointure entre le df_final et le df combinaisons pour corriger le problème d'omission de colonne lorsque charge = 0
       
        df_complet =pd.merge(df, combinaisons, on = ['jour', 'heure','site'], how = "right")
        df_complet['charge'].fillna(0, inplace=True)
       
        df = df_complet


# filtre le df sur la semaine suivante entière
        
        jour_deb = df['jour'].min().weekday()
        jour_a_ajouter = (7 - jour_deb) % 7
        deb_semaine_deux= df['jour'].min() + pd.Timedelta(days=jour_a_ajouter)
        
        fin_semaine_deux = deb_semaine_deux+ pd.Timedelta(days=6)
        
        df = df.loc[(df['jour'] >= deb_semaine_deux)&(df['jour']<= fin_semaine_deux)]
        
      
        df['jour']= pd.to_datetime(df['jour'])
        
       
        df = df.sort_values(by=['site', 'jour', 'heure'])

#transforme le dataframe dans un format "large"        
        
        df_pivot = df.pivot (index=['site', 'jour'], columns='heure', values='charge').reset_index() 
        
        
# calcul du cumul de charge horaire glissant à partir de données 10 min
        
        def calculate_sums(row): 
            sums = {}
            times = df_pivot.columns[2:]
            for i in range(len(times)-5):
                sum_value=sum(row[times[j]] for j in range (i,i+6))
                sums[times[i+5]] = sum_value
            return pd.Series(sums)
        
        new_df = df_pivot.apply (calculate_sums, axis=1)
            
        new_df.insert(0, 'site', df_pivot['site'])
        new_df.insert(1, 'jour', df_pivot['jour'])        
    
    # retourne le dataframe dans un format "long"
    
        df_depivote = pd.melt(new_df, id_vars=['site', 'jour'], var_name='heure', value_name='charge')

         
                                  
#new_df.to_excel("C:/Users/demanet/Downloads/test_cumul.xlsx", index= False)        
        
  
        
#créer une liste dynamique des sites (revient à faire unique(), mais permet ici de supprmier les sites fermées)
        df_config =df.groupby(['site'], as_index=False)['charge'].sum()
        df_config =df_config[df_config['charge'] > 0]
        df_config = df_config.drop(columns=['charge'])
        
                  
        custom_dict = {'K CTRCNT' : 0,
                       'K CTR' : 1,
                       'K CNT' : 2 , 
                       'L CTR' : 3, 
                       'L CNT' : 4, 
                       'M CTR' : 5, 
                       'Galerie EF' : 6, 
                       'C2F' : 7, 
                       'C2G' : 8, 
                       'Liaison AC' : 9,
                       'Liaison BD' : 10,
                        'T3': 11, 
                        'Terminal 1' : 12, 
                        'Terminal 1_5' : 13, 
                        'Terminal 1_6' : 14}
        
        df_config = df_config.sort_values(by=['site'], key=lambda x: x.map(custom_dict)).reset_index(drop=True)
    
        
        
        
        
        
        sites = df_config['site'].unique() # permet de créer un tableau simple à partir du dataframe df_complet (à revoir)
        
        
    # seuils de saturation    
        
        
        def seuil(site):
            seuils = {
                        'K CTRCNT' : 0,
                        'K CTR' : 1600,
                        'K CNT' : 300 , 
                        'L CTR' : 2100, 
                        'L CNT' :  1440, 
                        'M CTR' : 1820, 
                        'Galerie EF' : 1820, 
                        'C2F' : 2180, 
                        'C2G' : 910, 
                        'Liaison AC' : 1960,
                        'Liaison BD' : 2500,
                        'T3': 1260, 
                        'Terminal 1' : 2280, 
                        'Terminal 1_5' : 375, 
                        'Terminal 1_6' : 500
                        }
        
        
            return seuils.get(site,0)
        
    
    # graphique   
    
        for site in sites: 
            
            st.subheader(f" {site}")
            
            st.write("seuil max: ", seuil(site))
            df_site = df_depivote.loc[df_depivote['site']==site]
            locale.setlocale(locale.LC_ALL, 'fr_FR')
            df_site['Date']= df_site['jour'].dt.strftime('%A %d %b')
           
            fig = px.line(df_site, x= 'heure', y= 'charge',  color = 'Date',
                            labels={'jour', 'date'})
            
            ligne_seuil = seuil(site)
            fig.add_hline(y=ligne_seuil, line_dash='dash', line_color="red")
            
            
            
            st.plotly_chart(fig)
