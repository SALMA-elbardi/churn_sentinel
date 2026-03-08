import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CHARGEMENT DES FICHIERS ---
# Ajout de cache pour plus de fluidité
@st.cache_resource
def load_assets():
    model = joblib.load('model_churn.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Churn Sentinel Pro", layout="wide")

# --- CSS 
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    /* Style du bouton */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #004aad;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #003580;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Style des métriques natives */
    div[data-testid="stMetricValue"] { font-size: 38px; font-weight: 700; color: #1a1a1a; }
    div[data-testid="stMetricLabel"] { font-size: 16px; color: #5e5e5e; }
    
    # /* Style de la carte de résultat */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Analyseur de Rétention Client ")
st.markdown("---")

# --- FORMULAIRE DE SAISIE (SIDEBAR) ---
st.sidebar.header("📋 Profil du Client")

def user_input_features():
    tenure = st.sidebar.slider("Ancienneté (mois)", 1, 72, 24)
    monthly_charges = st.sidebar.slider("Facture Mensuelle ($)", 18.0, 120.0, 65.0)
    total_charges = tenure * monthly_charges 
    
    internet = st.sidebar.selectbox("Service Internet", ["DSL", "Fiber optic", "No"])
    contract = st.sidebar.selectbox("Type de Contrat", ["Month-to-month", "One year", "Two year"])
    payment = st.sidebar.selectbox("Mode de Paiement", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    security = st.sidebar.selectbox("Sécurité en ligne", ["No", "Yes", "No internet service"])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService': internet,
        'Contract': contract,
        'PaymentMethod': payment,
        'OnlineSecurity': security
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Sidebar footer pro
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Telecom Analytics | Développé par Salma")

def preprocess_input(df):
    processed_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Scaling
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    for col in numeric_cols:
        processed_df[col] = df[col]
        
    # Encodage catégoriel
    for col in ['InternetService', 'Contract', 'PaymentMethod', 'OnlineSecurity']:
        val = df[col][0]
        col_name = f"{col}_{val}"
        if col_name in model_columns:
            processed_df[col_name] = 1
            
    return processed_df

# --- ZONE D'AFFICHAGE PRINCIPALE ---
# On pré-enroule les colonnes dans un container pour le style
container = st.container()

# --- BOUTON D'ACTION ---
col_btn, _ = st.columns([1, 2])
with col_btn:
    predict_btn = st.button("🔍 Analyser le risque")

if predict_btn:
    final_input = preprocess_input(input_df)
    
    # Calcul des probabilités
    proba = model.predict_proba(final_input)[0][1]
    risk_percentage = proba * 100
    
    # Définition des couleurs pro selon le risque
    if risk_percentage > 70:
        status_color, bar_color = "#e74c3c", "#e74c3c" # Rouge
    elif risk_percentage > 35:
        status_color, bar_color = "#f39c12", "#f1c40f" # Orange
    else:
        status_color, bar_color = "#27ae60", "#2ecc71" # Vert

  
    col1, col2 = st.columns([1, 1.5])

   
    with col1:
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("### ⚖️ Score de Risque")
        

        st.metric(label="Probabilité de désabonnement", value=f"{risk_percentage:.1f}%")
        
        # 2. Barre de progression CSS fine et épurée 
        st.markdown(f"""
            <div style="background-color: #e0e0e0; border-radius: 10px; width: 100%; height: 8px; margin-top: 10px;">
                <div style="background-color: {bar_color}; width: {risk_percentage}%; height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out;"></div>
            </div>
            <p style="color: {status_color}; font-weight: bold; margin-top: 10px; font-size: 14px;">
                Statut : {'Critique' if risk_percentage > 70 else 'A surveiller' if risk_percentage > 35 else 'Stable'}
            </p>
            """, unsafe_allow_html=True)
        
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLONNE 2 : DIAGNOSTIC ---
    with col2: 
        st.write("###  Diagnostic et Stratégie")
        
        if proba > 0.7:
            st.error("**ALERTE : Score de churn élevé .**")
            st.subheader("🎯 Actions Prescriptives :")
            
            
            with st.expander("Voir les recommandations détaillées", expanded=True):
                if input_df['Contract'].iloc[0] == "Month-to-month":
                    st.write("👉 **Fidélisation :** Le client est sans engagement. Proposez une conversion vers un contrat annuel avec une remise de 15% pour sécuriser le revenu.")
                
                if input_df['OnlineSecurity'].iloc[0] == "No":
                    st.write("🛠️ **Upsell Protecteur :** Offrez 3 mois gratuits de 'Online Security'. Les données montrent que ce service réduit le churn.")
                    
                if input_df['MonthlyCharges'].iloc[0] > 70:
                    st.write("💰 **Optimisation Tarifaire :** Les charges mensuelles sont élevées. Proposez un Bundle (pack groupé) pour améliorer la perception de valeur.")

        elif proba > 0.4:
            st.warning("**AVERTISSEMENT : Profil instable.**")
            st.subheader("📞 Actions Préventives :")
            st.write("1. Planifiez un appel de satisfaction  sous 48h.")
            st.write("2. Envoyez une offre de fidélité ciblée sur ses services actuels.")
            
        else:
            st.success("**CONFIRMATION : Profil client stable.**")
            st.subheader(" Stratégie de Croissance :")

            st.write("Le client est satisfait. Invitez-le à parrainer un nouveau client ou à laisser un avis positif.")
