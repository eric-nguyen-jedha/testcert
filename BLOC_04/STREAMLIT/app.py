import streamlit as st
import pandas as pd
import boto3
import folium
from streamlit_folium import st_folium
from io import StringIO
from babel.dates import format_datetime
import os
import time

# --- Configuration de la page (à faire en premier) ---
st.set_page_config(layout="wide")

# --- CSS pour élargir le canevas ---
st.markdown("""
    <style>
        .stApp > header, .stApp > footer { visibility: hidden; }
        .block-container { max-width: 1400px !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    </style>
""", unsafe_allow_html=True)
st.title("🌦️ Météo France")

# --- Secrets et connexion S3 (ne changent pas) ---
@st.cache_resource
def get_s3_client():
    required_secrets = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    if not all(secret in os.environ for secret in required_secrets):
        st.error("⚠️ Secrets AWS manquants.")
        st.stop()

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-west-3")
    )
    return s3

s3 = get_s3_client()
BUCKET = "certiflead"

# --- URL de base GitHub ---
GITHUB_URL = "https://raw.githubusercontent.com/eric-nguyen-jedha/AIA/eda19edbeea136fc3b9d1f966237c73a9a2c50c1/BLOC_04/img/"

# --- Mapping des prédictions vers les noms de fichiers ---
icon_names = {
    "Clear": "clear",
    "Clouds": "cloud",
    "Rain": "rain",
    "Fog": "fog",
}

# --- Fonction de génération (sans aucun décorateur) ---
def generate_maps():
    """Charge les données et crée les objets cartes Folium."""
    print(f"[{time.strftime('%H:%M:%S')}] Exécution de generate_maps: chargement des données et création des cartes.")

    # Fonctions internes pour charger les données
    def read_csv(key):
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")), sep=",")
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
            df["timestamp_local"] = df["timestamp"].dt.tz_convert("Europe/Paris")
            df["timestamp_display"] = df["timestamp_local"].dt.strftime("%d/%m %H:%M")
            return df
        except Exception as e:
            st.warning(f"Impossible de charger {key}: {e}")
            return pd.DataFrame()

    df_hist = read_csv("historical.csv")
    df_fore = read_csv("forecast_6h.csv")

    if df_hist.empty and df_fore.empty:
        return None

    # Carte Historique
    m_hist = folium.Map(location=[46.6, 2.2], zoom_start=6, tiles="USGS.USImagery")

    if not df_hist.empty:
        # Pré-calculer les URLs des icônes AVANT la boucle
        df_hist['icon_url'] = df_hist.apply(
            lambda r: f"{GITHUB_URL}{icon_names.get(r.prediction, 'clear')}-{'day' if 9 <= r.timestamp_local.hour < 18 else 'night'}.svg",
            axis=1
        )

        for _, r in df_hist.iterrows():
            folium.Marker(
                [r.lat, r.lon],
                popup=f"🏙️ {r.ville}<br><b>Now :</b> {r.prediction}<br>{r.timestamp_display}",
                icon=folium.CustomIcon(r.icon_url, icon_size=(55, 55))
            ).add_to(m_hist)
            # Ajouter une étiquette avec le nom de la ville
            folium.map.Marker(
                [r.lat + 0.5, r.lon],  # Décaler légèrement pour éviter la superposition
                icon=folium.DivIcon(
                    html=f"""
                    <div style="font-family: Arial; font-size: 12pt; color: yellow">
                        {r.ville}
                    </div>
                    """
                )
            ).add_to(m_hist)

    # Carte Prévision
    m_fore = folium.Map(location=[46.6, 2.2], zoom_start=6, tiles="USGS.USImagery")

    if not df_fore.empty:
        # Pré-calculer les URLs des icônes AVANT la boucle
        df_fore['icon_url'] = df_fore.apply(
            lambda r: f"{GITHUB_URL}{icon_names.get(r.prediction, 'clear')}-{'day' if 9 <= r.timestamp_local.hour < 18 else 'night'}.svg",
            axis=1
        )

        for _, r in df_fore.iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=10, color="orange", fill=True, fill_opacity=0.25).add_to(m_fore)
            folium.Marker(
                [r.lat, r.lon],
                popup=f"🏙️ {r.ville}<br><b>Forecast +6h :</b> {r.prediction}<br>{r.timestamp_display}",
                icon=folium.CustomIcon(r.icon_url, icon_size=(55, 55))
            ).add_to(m_fore)
            # Ajouter une étiquette avec le nom de la ville
            folium.map.Marker(
                [r.lat + 0.5, r.lon],  # Décaler légèrement pour éviter la superposition
                icon=folium.DivIcon(
                    html=f"""
                    <div style="font-family: Arial; font-size: 12pt; color: yellow;">
                        {r.ville}
                    </div>
                    """
                )
            ).add_to(m_fore)

    # Date
    timestamps = [ts for ts in [df_hist['timestamp_local'].max(), df_fore['timestamp_local'].max()] if pd.notna(ts)]
    latest_ts = max(timestamps) if timestamps else None

    return {"hist": m_hist, "fore": m_fore, "ts": latest_ts}

# --- Logique principale de l'application ---
# Initialisation de l'état si nécessaire
if 'last_map_update' not in st.session_state:
    st.session_state.last_map_update = 0
    st.session_state.maps_data = None

# Vérifier si on doit mettre à jour les cartes (plus de 300 secondes)
should_update = time.time() - st.session_state.last_map_update > 300

if should_update:
    st.session_state.maps_data = generate_maps()
    st.session_state.last_map_update = time.time()

# Affichage des cartes depuis le session_state
if st.session_state.maps_data:
    maps = st.session_state.maps_data

    st.subheader(f"{format_datetime(maps['ts'], 'd MMMM y à HH:mm', locale='fr_FR')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🕙 Now")
        st_folium(maps['hist'], key="hist_map", use_container_width=True, height=550, returned_objects=[])
    with col2:
        st.markdown("### 🔮 Forecast +6h")
        st_folium(maps['fore'], key="fore_map", use_container_width=True, height=550, returned_objects=[])
else:
    st.warning("Aucune donnée disponible pour générer les cartes. Prochaine tentative dans 5 minutes.")