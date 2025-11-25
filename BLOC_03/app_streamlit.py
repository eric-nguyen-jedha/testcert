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
        ["🏠 Dashboard", "🚨 Fraudes (24h)", "✅ Non Fraudes (24h)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ℹ️ À propos
    Dashboard de détection de fraude en temps réel.
    
    **🔄 Refresh** : Cliquez sur le bouton pour charger/actualiser les données.
    
    **⚡ Optimisé** : Les données ne se chargent que sur demande pour économiser les ressources.
    
    **📊 Données** : Dernières 24h pour les pages de détail.
    """)
    
    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🚨 Fraudes (24h)":
        page_frauds()
    elif page == "✅ Non Fraudes (24h)":
        page_no_frauds()

if __name__ == "__main__":
    main()