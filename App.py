import streamlit as st
import joblib 
import sklearn
import numpy as np


# charger le model
model_lr = joblib.load('model_lr.jbl')
# Charger le scaler sauvegardé



# Titre de l'Application
st.title("Prédiction des Prix de l'Immobilier")
st.write("Entrez les caractéristiques de la maison pour prédire le prix: ")

with st.form("Formulaire"):
    LotFrontage = st.number_input("Pieds linéaire de rue", min_value=20)
    LotArea = st.number_input("Taille du terrain en pieds carrés", min_value=100)
    OverallQual = st.number_input("Qualité générale", min_value=1, max_value=10)
    OverallCond = st.number_input("État général", min_value=1, max_value=10)
    YearBuilt = st.number_input("Année de construction originale", min_value=1800, max_value=2024)
    YearRemodAdd = st.number_input("Année de rénovation", min_value=1900, max_value=2024)
    BsmtFinSF1 = st.number_input("Pieds carrés finis de type 1", min_value=30)
    BsmtUnfSF = st.number_input("Pieds carrés non finis du sous-sol", min_value=30)
    TotalBsmtSF = st.number_input("Superficie totale du sous-sol en pieds carrés", min_value=30)
    CentralAir = st.selectbox("Climatisation centrale", options=['Y', 'N'])
    FirstFlrSF = st.number_input("Superficie du rez-de-chaussée", min_value=10)
    SecondFlrSF = st.number_input("Superficie du deuxième étage", min_value=10)
    GrLivArea = st.number_input("Superficie habitable hors-sol", min_value=10)
    Fireplaces = st.number_input("Nombre de cheminées", min_value=0)
    GarageCars = st.number_input("Capacité du garage (en voitures)", min_value=0, max_value=10)
    GarageArea = st.number_input("Superficie du garage en pieds carrés", min_value=10)
    GarageFinish = st.selectbox("Finition intérieure du garage", options=['Unf', 'RFn', 'Fin'])
    
    valider = st.form_submit_button("Envoyer")

if valider:
        CentralAir = 1 if CentralAir == 'Y' else 0
        GarageFinish = 1 if GarageFinish == 'RFn' else 0
        
        features = np.array([
            LotFrontage, LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd,
            BsmtFinSF1, BsmtUnfSF, TotalBsmtSF, CentralAir, FirstFlrSF, SecondFlrSF,
            GrLivArea, Fireplaces, GarageCars, GarageArea, GarageFinish
        ]).reshape(1, -1)


        # Prédire le prix
        predicted_price=model_lr.predict(features)[0]

        # Afficher le résultat
        st.write(f'Prix prédit : {predicted_price:,.2f}')
