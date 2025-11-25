import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os

# ========================== CONFIGURATION ==========================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵🏻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Couleurs (variables pour personnalisation facile)
COLOR_FRAUD = "#FF4B4B"      # Rouge
COLOR_NO_FRAUD = "#00CC66"   # Vert
COLOR_SAVED = "#FFD700"      # Or pour l'argent économisé

# ========================== CONNEXION BASE DE DONNÉES ==========================
@st.cache_resource
def get_db_connection():
    """Connexion directe à Neon DB (non-pooler, stable pour petit volume)"""
    try:
        database_url = os.environ.get("NEON_DB_FRAUD_URL")
        if not database_url:
            st.error("❌ Variable NEON_DB_FRAUD_URL non trouvée dans les secrets Hugging Face")
            st.stop()
        
        # Connexion directe sans pooler → autorise les options PostgreSQL
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"
            }
        )
        return engine
    except Exception as e:
        st.error(f"❌ Erreur de connexion à la base de données: {e}")
        st.stop()


# ========================== CHARGEMENT DES DONNÉES ==========================
@st.cache_data(ttl=3600)
def load_csv_data():
    """Charge le fichier CSV pour l'EDA"""
    try:
        df = pd.read_csv("fraudTest.csv")
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier CSV: {e}")
        return pd.DataFrame()


# ========================== REQUÊTES SQL OPTIMISÉES ==========================
def load_all_data():
    """Charge toutes les transactions - APPELÉ SEULEMENT APRÈS CLIC SUR REFRESH"""
    engine = get_db_connection()
    query = text("""
        SELECT 
            trans_num, merchant, category, amt, gender, city, zip, city_pop, job,
            hour, day, month, year, pred_is_fraud, is_fraud_ground_truth,
            transaction_time, created_at
        FROM fraud_predictions
        ORDER BY created_at DESC
        LIMIT 10000
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

def load_last_24h_data():
    """Charge les transactions des dernières 24h"""
    engine = get_db_connection()
    query = text("""
        SELECT trans_num, merchant, category, amt, gender, city, pred_is_fraud, created_at
        FROM fraud_predictions
        WHERE created_at >= NOW() - INTERVAL '24 HOURS'
        ORDER BY created_at DESC
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données 24h: {e}")
        return pd.DataFrame()

def load_last_7_days_stats():
    """Charge les stats agrégées des 7 derniers jours"""
    engine = get_db_connection()
    query = text("""
        SELECT DATE(created_at) as date,
               SUM(CASE WHEN pred_is_fraud = 1 THEN 1 ELSE 0 END) as frauds,
               SUM(CASE WHEN pred_is_fraud = 0 THEN 1 ELSE 0 END) as no_frauds
        FROM fraud_predictions
        WHERE created_at >= NOW() - INTERVAL '7 DAYS'
        GROUP BY DATE(created_at)
        ORDER BY date ASC
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des stats 7 jours: {e}")
        return pd.DataFrame()

def load_dashboard_summary():
    """Charge les métriques pour le dashboard (robuste à toutes versions SQLAlchemy)"""
    engine = get_db_connection()
    query = text("""
        SELECT 
            COUNT(*) as total_transactions,
            COALESCE(SUM(CASE WHEN pred_is_fraud = 1 THEN 1 ELSE 0 END), 0) as total_frauds,
            COALESCE(SUM(CASE WHEN pred_is_fraud = 0 THEN 1 ELSE 0 END), 0) as total_no_frauds,
            COALESCE(SUM(CASE WHEN pred_is_fraud = 1 THEN amt ELSE 0 END), 0) as total_fraud_amount
        FROM fraud_predictions;
    """)

    try:
        with engine.connect() as conn:
            # mappins().first() renvoie un dict-like (compatible SQLAlchemy 1.x/2.x)
            result = conn.execute(query).mappings().first()

        if not result:
            return {'total_frauds': 0, 'total_no_frauds': 0, 'total_fraud_amount': 0.0}

        # Convertir explicitement en float pour éviter Decimal * float errors
        total_fraud_amount = result.get('total_fraud_amount', 0) or 0
        try:
            total_fraud_amount = float(total_fraud_amount)
        except (TypeError, ValueError):
            total_fraud_amount = 0.0

        return {
            'total_frauds': int(result.get('total_frauds', 0) or 0),
            'total_no_frauds': int(result.get('total_no_frauds', 0) or 0),
            'total_fraud_amount': total_fraud_amount
        }

    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du résumé: {e}")
        return {'total_frauds': 0, 'total_no_frauds': 0, 'total_fraud_amount': 0.0}



# ========================== PAGE: DASHBOARD ==========================
def page_dashboard():
    st.title("🕵🏻 Fraud Detection Dashboard")
    
    # Message d'instruction
    st.info("👇 Cliquez sur **Refresh Data** pour charger les données")
    
    # Bouton refresh qui contrôle le chargement
    if st.button("🔄 Refresh Data", type="primary", key="dashboard_refresh"):
        st.session_state.dashboard_loaded = True
    
    # Ne charger que si le bouton a été cliqué
    if not st.session_state.get('dashboard_loaded', False):
        st.warning("⚠️ Cliquez sur 'Refresh Data' pour afficher le dashboard")
        return
    
    with st.spinner("Chargement des données..."):
        # Charger d'abord le résumé (rapide)
        summary = load_dashboard_summary()
        df_7days = load_last_7_days_stats()
    
    # ========================== MÉTRIQUES ==========================
    total_frauds = summary['total_frauds']
    total_no_frauds = summary['total_no_frauds']
    saved_amount = int(summary['total_fraud_amount'] * 1.5)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background-color: {COLOR_FRAUD}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">🚨 Frauds</h3>
            <h1 style="color: white; margin: 10px 0;">{total_frauds}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color: {COLOR_NO_FRAUD}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">✅ No Frauds</h3>
            <h1 style="color: white; margin: 10px 0;">{total_no_frauds}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background-color: {COLOR_SAVED}; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">💰 Saved Amount</h3>
            <h1 style="color: white; margin: 10px 0;">${saved_amount}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================== GRAPHIQUES ==========================
    col_pie, col_saved_detail = st.columns([1, 1])
    
    with col_pie:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Frauds', 'No Frauds'],
            values=[total_frauds, total_no_frauds],
            marker=dict(colors=[COLOR_FRAUD, COLOR_NO_FRAUD]),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=14
        )])
        fig_pie.update_layout(title="Distribution Fraud vs No Fraud", showlegend=True, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_saved_detail:
        total_fraud_amount = summary['total_fraud_amount']
        additional_costs = total_fraud_amount * 0.5
        st.markdown("### 💵 Breakdown of Saved Amount")
        st.markdown(f"""
        - **Total Fraud Amounts**: ${total_fraud_amount:,.2f}
        - **Estimated Additional Costs** (chargebacks, fees): ${additional_costs:,.2f}
        - **Total Saved**: ${saved_amount:,.2f}
        """)
        fig_breakdown = go.Figure(data=[
            go.Bar(name='Fraud Amount', x=['Saved'], y=[total_fraud_amount], marker_color=COLOR_FRAUD),
            go.Bar(name='Additional Costs', x=['Saved'], y=[additional_costs], marker_color=COLOR_SAVED)
        ])
        fig_breakdown.update_layout(barmode='stack', showlegend=True, height=250, yaxis_title="Amount ($)")
        st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # ========================== GRAPHIQUE 7 JOURS ==========================
    st.markdown("### 📊 Fraud Trend - Last 7 Days")
    if not df_7days.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(name='Frauds', x=df_7days['date'], y=df_7days['frauds'], marker_color=COLOR_FRAUD))
        fig_trend.add_trace(go.Bar(name='No Frauds', x=df_7days['date'], y=df_7days['no_frauds'], marker_color=COLOR_NO_FRAUD))
        fig_trend.update_layout(barmode='stack', xaxis_title="Date", yaxis_title="Number of Transactions", height=400, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Pas encore de données sur 7 jours")


# ========================== PAGE: EDA ==========================
def page_eda():
    st.title("📊 Exploratory Data Analysis")
    
    st.info("👇 Cliquez sur **Load Data** pour charger les données EDA")
    
    if st.button("🔄 Load Data", type="primary", key="eda_refresh"):
        st.session_state.eda_loaded = True
    
    if not st.session_state.get('eda_loaded', False):
        st.warning("⚠️ Cliquez sur 'Load Data' pour afficher l'analyse")
        return
    
    with st.spinner("Chargement des données..."):
        df = load_csv_data()
    
    if df.empty:
        st.error("Impossible de charger les données")
        return
    
    # ========================== 1. RÉSUMÉ DU DATASET ==========================
    st.markdown("## 📋 Résumé du Dataset")
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Nombre de lignes", f"{len(df):,}")
    with col2:
        st.metric("📋 Nombre de colonnes", f"{len(df.columns)}")
    with col3:
        st.metric("💾 Taille mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("🔄 Doublons", f"{duplicates:,}")
    
    # Valeurs manquantes
    st.markdown("### 🔍 Valeurs manquantes")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Colonne': missing.index,
        'Manquantes': missing.values,
        'Pourcentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Manquantes'] > 0].sort_values('Manquantes', ascending=False)
    
    if not missing_df.empty:
        fig_missing = px.bar(
            missing_df,
            x='Colonne',
            y='Pourcentage',
            title='Pourcentage de valeurs manquantes par colonne',
            color='Pourcentage',
            color_continuous_scale='Reds',
            text=missing_df['Pourcentage'].apply(lambda x: f"{x:.1f}%")
        )
        fig_missing.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("✅ Aucune valeur manquante dans le dataset !")
    
    # Statistiques descriptives
    st.markdown("### 📊 Statistiques descriptives (Variables numériques)")
    
    # Sélecteur de colonnes numériques
    numeric_cols_all = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_stats_cols = st.multiselect(
        "Choisissez les colonnes à analyser",
        numeric_cols_all,
        default=numeric_cols_all[:5]
    )
    
    if selected_stats_cols:
        stats_df = df[selected_stats_cols].describe().T
        stats_df['missing'] = df[selected_stats_cols].isnull().sum().values
        stats_df['missing_pct'] = (stats_df['missing'] / len(df) * 100).round(2)
        
        # Formater pour l'affichage
        display_stats = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing', 'missing_pct']]
        display_stats.columns = ['Count', 'Moyenne', 'Écart-type', 'Min', 'Q1', 'Médiane', 'Q3', 'Max', 'Manquantes', 'Manquantes (%)']
        
        st.dataframe(
            display_stats.style.format({
                'Moyenne': '{:.2f}',
                'Écart-type': '{:.2f}',
                'Min': '{:.2f}',
                'Q1': '{:.2f}',
                'Médiane': '{:.2f}',
                'Q3': '{:.2f}',
                'Max': '{:.2f}',
                'Manquantes (%)': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Distribution des variables numériques
        st.markdown("### 📈 Distributions des variables numériques")
        selected_dist = st.selectbox("Choisissez une variable à visualiser", selected_stats_cols)
        
        col_hist, col_box = st.columns(2)
        
        with col_hist:
            fig_hist = px.histogram(
                df,
                x=selected_dist,
                nbins=50,
                title=f"Distribution de {selected_dist}",
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_box:
            fig_box = px.box(
                df,
                y=selected_dist,
                title=f"Box plot de {selected_dist}",
                color_discrete_sequence=['#636EFA']
            )
            fig_box.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Variables catégorielles
    st.markdown("### 🏷️ Variables catégorielles")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        selected_cat = st.selectbox("Choisissez une variable catégorielle", categorical_cols)
        
        value_counts = df[selected_cat].value_counts().head(15)
        
        col_bar, col_info = st.columns([2, 1])
        
        with col_bar:
            fig_cat = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Top 15 valeurs de {selected_cat}",
                labels={'x': selected_cat, 'y': 'Count'},
                color=value_counts.values,
                color_continuous_scale='Blues'
            )
            fig_cat.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col_info:
            st.markdown("#### Statistiques")
            st.metric("Valeurs uniques", df[selected_cat].nunique())
            st.metric("Valeur la plus fréquente", value_counts.index[0])
            st.metric("Fréquence max", f"{value_counts.values[0]:,}")
            st.metric("% de la plus fréquente", f"{(value_counts.values[0] / len(df) * 100):.1f}%")
    else:
        st.info("Aucune variable catégorielle détectée")
    
    st.markdown("---")
    
    # ========================== 2. DISTRIBUTION FRAUDE vs NON-FRAUDE ==========================
    st.markdown("## 🥧 Distribution des transactions")
    
    fraud_counts = df["is_fraud"].value_counts().reset_index()
    fraud_counts.columns = ["is_fraud", "count"]
    fraud_counts["label"] = fraud_counts["is_fraud"].map({0: "Non frauduleuse", 1: "Frauduleuse"})
    
    fig_pie = px.pie(
        fraud_counts,
        values="count",
        names="label",
        title="Répartition des transactions : frauduleuses vs non frauduleuses",
        color_discrete_sequence=["#636EFA", "#EF553B"],
        hole=0.4
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Statistiques
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraudes", f"{fraud_counts[fraud_counts['is_fraud']==1]['count'].values[0]:,}")
    with col3:
        fraud_rate = (fraud_counts[fraud_counts['is_fraud']==1]['count'].values[0] / len(df)) * 100
        st.metric("Taux de fraude", f"{fraud_rate:.2f}%")
    
    st.markdown("---")
    
    # ========================== 3. CARTE GÉOGRAPHIQUE ==========================
    st.markdown("## 🗺️ Localisation géographique des transactions")
    
    # Vérifier si les colonnes existent
    if 'merch_lat' in df.columns and 'merch_long' in df.columns:
        df_geo = df.dropna(subset=["merch_lat", "merch_long"])
        
        # Option d'échantillonnage pour performance
        sample_size = st.slider("Nombre de points à afficher", 1000, min(50000, len(df_geo)), 10000, step=1000)
        df_sample = df_geo.sample(n=min(sample_size, len(df_geo)), random_state=42)
        
        # Ajouter le label
        df_sample["fraud_label"] = df_sample["is_fraud"].map({0: "Non frauduleuse", 1: "Frauduleuse"})
        
        fig_map = px.scatter_mapbox(
            df_sample,
            lat="merch_lat",
            lon="merch_long",
            color="fraud_label",
            color_discrete_map={"Non frauduleuse": "#636EFA", "Frauduleuse": "#EF553B"},
            title=f"Localisation des transactions ({sample_size} points échantillonnés)",
            mapbox_style="open-street-map",
            zoom=3,
            height=700,
            hover_data=["amt", "category", "merchant"]
        )
        
        fig_map.update_layout(
            legend_title_text="Type de transaction",
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("⚠️ Les colonnes de géolocalisation (merch_lat, merch_long) ne sont pas disponibles dans le dataset")
    
    st.markdown("---")
    
    # ========================== 4. FRAUDES PAR GENRE ==========================
    st.markdown("## 👥 Analyse par genre")
    
    if 'gender' in df.columns:
        # Nombre de fraudes par genre
        fraud_by_gender = df[df["is_fraud"] == 1]["gender"].value_counts().reset_index()
        fraud_by_gender.columns = ["gender", "count"]
        fraud_by_gender["gender_label"] = fraud_by_gender["gender"].map({"M": "Homme", "F": "Femme"})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gender = px.bar(
                fraud_by_gender,
                x="gender_label",
                y="count",
                color="gender_label",
                color_discrete_map={"Homme": "#1f77b4", "Femme": "#ff7f0e"},
                title="Nombre de fraudes par genre",
                labels={"count": "Nombre de fraudes", "gender_label": "Genre"},
                text="count"
            )
            fig_gender.update_layout(showlegend=False)
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            # Taux de fraude par genre
            gender_stats = df.groupby('gender')['is_fraud'].agg(['sum', 'count']).reset_index()
            gender_stats['fraud_rate'] = (gender_stats['sum'] / gender_stats['count']) * 100
            gender_stats['gender_label'] = gender_stats['gender'].map({"M": "Homme", "F": "Femme"})
            
            fig_rate = px.bar(
                gender_stats,
                x="gender_label",
                y="fraud_rate",
                color="gender_label",
                color_discrete_map={"Homme": "#1f77b4", "Femme": "#ff7f0e"},
                title="Taux de fraude par genre (%)",
                labels={"fraud_rate": "Taux de fraude (%)", "gender_label": "Genre"},
                text=gender_stats['fraud_rate'].apply(lambda x: f"{x:.2f}%")
            )
            fig_rate.update_layout(showlegend=False)
            st.plotly_chart(fig_rate, use_container_width=True)
    else:
        st.warning("⚠️ La colonne 'gender' n'est pas disponible dans le dataset")
    
    st.markdown("---")
    
    # ========================== 5. CORRÉLATIONS ET PAIRPLOT ==========================
    st.markdown("## 🔍 Analyse de corrélations")
    
    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Retirer is_fraud de la liste
    if 'is_fraud' in numeric_cols:
        numeric_cols.remove('is_fraud')
    
    # Limiter aux colonnes les plus pertinentes par défaut
    default_cols = ['amt', 'city_pop', 'lat', 'long'][:4]
    default_cols = [col for col in default_cols if col in numeric_cols]
    
    selected_cols = st.multiselect(
        "Choisissez les variables à analyser (3-5 variables recommandées)",
        numeric_cols,
        default=default_cols[:4],
        max_selections=6
    )
    
    if selected_cols and len(selected_cols) >= 2:
        # ========================== MATRICE DE CORRÉLATION ==========================
        st.markdown("### 📊 Matrice de corrélation")
        
        corr_matrix = df[selected_cols].corr()
        
        # Créer une heatmap annotée plus lisible
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title="Corrélation")
        ))
        
        fig_corr.update_layout(
            title="Matrice de corrélation de Pearson",
            xaxis_title="",
            yaxis_title="",
            height=500,
            width=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Afficher les corrélations les plus fortes
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Corrélation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Corrélation', key=abs, ascending=False)
        
        st.markdown("#### 🔝 Top 5 des corrélations")
        st.dataframe(
            corr_df.head(5).style.background_gradient(
                subset=['Corrélation'], 
                cmap='RdBu_r', 
                vmin=-1, 
                vmax=1
            ).format({'Corrélation': '{:.3f}'}),
            use_container_width=True
        )
        
        # ========================== PAIRPLOT STYLE SEABORN ==========================
        st.markdown("### 🎨 Pairplot (style Seaborn)")
        
        # Échantillonner pour la performance
        sample_size_pair = min(1000, len(df))
        df_pair = df[selected_cols + ['is_fraud']].sample(n=sample_size_pair, random_state=42)
        df_pair['fraud_label'] = df_pair['is_fraud'].map({0: "Non frauduleuse", 1: "Frauduleuse"})
        
        # Créer une grille de subplots
        n_vars = len(selected_cols)
        
        from plotly.subplots import make_subplots
        
        fig_pair = make_subplots(
            rows=n_vars, 
            cols=n_vars,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.02,
            horizontal_spacing=0.02
        )
        
        colors = {'Non frauduleuse': '#636EFA', 'Frauduleuse': '#EF553B'}
        
        for i, var_y in enumerate(selected_cols):
            for j, var_x in enumerate(selected_cols):
                row = i + 1
                col = j + 1
                
                if i == j:
                    # Diagonale : histogrammes
                    for fraud_label in ['Non frauduleuse', 'Frauduleuse']:
                        data = df_pair[df_pair['fraud_label'] == fraud_label][var_x]
                        fig_pair.add_trace(
                            go.Histogram(
                                x=data,
                                name=fraud_label,
                                marker_color=colors[fraud_label],
                                opacity=0.7,
                                showlegend=(i == 0 and j == 0)
                            ),
                            row=row, col=col
                        )
                elif i > j:
                    # Triangle inférieur : scatter plots
                    for fraud_label in ['Non frauduleuse', 'Frauduleuse']:
                        data = df_pair[df_pair['fraud_label'] == fraud_label]
                        fig_pair.add_trace(
                            go.Scatter(
                                x=data[var_x],
                                y=data[var_y],
                                mode='markers',
                                name=fraud_label,
                                marker=dict(
                                    color=colors[fraud_label],
                                    size=4,
                                    opacity=0.6
                                ),
                                showlegend=False
                            ),
                            row=row, col=col
                        )
                
                # Labels uniquement sur les bords
                if i == n_vars - 1:
                    fig_pair.update_xaxes(title_text=var_x, row=row, col=col)
                if j == 0:
                    fig_pair.update_yaxes(title_text=var_y, row=row, col=col)
        
        fig_pair.update_layout(
            height=200 * n_vars,
            title_text=f"Pairplot - {sample_size_pair} échantillons",
            showlegend=True
        )
        
        st.plotly_chart(fig_pair, use_container_width=True)
        
        st.info("💡 **Lecture du pairplot** : La diagonale montre la distribution de chaque variable. Le triangle inférieur montre les relations entre paires de variables.")
        
    elif selected_cols and len(selected_cols) < 2:
        st.warning("⚠️ Veuillez sélectionner au moins 2 variables pour voir les corrélations")
    else:
        st.warning("⚠️ Veuillez sélectionner des variables à analyser")


# ========================== PAGE: FRAUDES (24h) ==========================
def page_frauds():
    st.title("🚨 Fraudes Détectées (Dernières 24h)")
    
    st.info("👇 Cliquez sur **Refresh Data** pour charger les fraudes")
    
    if st.button("🔄 Refresh Data", type="primary", key="frauds_refresh"):
        st.session_state.frauds_loaded = True
    
    if not st.session_state.get('frauds_loaded', False):
        st.warning("⚠️ Cliquez sur 'Refresh Data' pour afficher les fraudes")
        return
    
    with st.spinner("Chargement des fraudes..."):
        df = load_last_24h_data()
        df_frauds = df[df['pred_is_fraud'] == 1]
    
    st.markdown(f"""
    <div style="background-color: {COLOR_FRAUD}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🚨 {len(df_frauds)} Fraudes détectées</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not df_frauds.empty:
        st.dataframe(
            df_frauds[['trans_num','merchant','category','amt','city','gender','created_at']].sort_values('created_at', ascending=False),
            use_container_width=True,
            height=600
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Montant total", f"${df_frauds['amt'].sum():,.2f}")
        with col2:
            st.metric("Montant moyen", f"${df_frauds['amt'].mean():,.2f}")
        with col3:
            st.metric("Montant max", f"${df_frauds['amt'].max():,.2f}")
    else:
        st.success("✅ Aucune fraude détectée dans les dernières 24h !")

# ========================== PAGE: NON FRAUDES (24h) ==========================
def page_no_frauds():
    st.title("✅ Transactions Légitimes (Dernières 24h)")
    
    st.info("👇 Cliquez sur **Refresh Data** pour charger les transactions")
    
    if st.button("🔄 Refresh Data", type="primary", key="no_frauds_refresh"):
        st.session_state.no_frauds_loaded = True
    
    if not st.session_state.get('no_frauds_loaded', False):
        st.warning("⚠️ Cliquez sur 'Refresh Data' pour afficher les transactions")
        return
    
    with st.spinner("Chargement des transactions légitimes..."):
        df = load_last_24h_data()
        df_no_frauds = df[df['pred_is_fraud'] == 0]
    
    st.markdown(f"""
    <div style="background-color: {COLOR_NO_FRAUD}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">✅ {len(df_no_frauds)} Transactions légitimes</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not df_no_frauds.empty:
        st.dataframe(
            df_no_frauds[['trans_num','merchant','category','amt','city','gender','created_at']].sort_values('created_at', ascending=False),
            use_container_width=True,
            height=600
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Montant total", f"${df_no_frauds['amt'].sum():,.2f}")
        with col2:
            st.metric("Montant moyen", f"${df_no_frauds['amt'].mean():,.2f}")
        with col3:
            st.metric("Montant max", f"${df_no_frauds['amt'].max():,.2f}")
    else:
        st.warning("⚠️ Aucune transaction légitime dans les dernières 24h")

# ========================== NAVIGATION ==========================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Dashboard", "📊 EDA", "🚨 Fraudes (24h)", "✅ Non Fraudes (24h)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ℹ️ À propos
    Dashboard de détection de fraude en temps réel.
    
    **🔄 Refresh** : Cliquez sur le bouton pour charger/actualiser les données.
    
    **⚡ Optimisé** : Les données ne se chargent que sur demande pour économiser les ressources.
    
    **📊 Données** : 
    - Dashboard: Stats temps réel
    - EDA: Analyse du dataset complet
    - Détail: Dernières 24h
    """)
    
    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "📊 EDA":
        page_eda()
    elif page == "🚨 Fraudes (24h)":
        page_frauds()
    elif page == "✅ Non Fraudes (24h)":
        page_no_frauds()

if __name__ == "__main__":
    main()