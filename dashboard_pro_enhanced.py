"""
لوحة تحكم الري الذكية - Tableau de Bord d'Irrigation Intelligent
Interface Professionnelle Bilingue (FR/AR) avec Prédiction en Temps Réel
VERSION FINALE CORRIGÉE - Compatible avec edge_model.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import joblib
import os

# ===== CONFIGURATION DE LA PAGE =====
st.set_page_config(
    page_title="💧 Irrigation Intelligente",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== STYLES CSS PROFESSIONNELS =====
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
    }
    
    .header-title {
        text-align: center;
        color: #1b5e20;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        text-align: center;
        color: #555;
        font-size: 1em;
        margin-bottom: 30px;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 6px solid #2e7d32;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9em;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    
    .stat-value {
        color: #1b5e20;
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .stat-unit {
        color: #999;
        font-size: 0.9em;
    }
    
    .zone-card {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .zone-card h3 {
        font-size: 1.5em;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }
    
    .zone-stat {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }
    
    .zone-label {
        font-weight: 600;
    }
    
    .zone-value {
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .status-ok {
        background: #c8e6c9;
        color: #1b5e20;
        padding: 3px 8px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .status-warning {
        background: #fff9c4;
        color: #f57f17;
        padding: 3px 8px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .status-critical {
        background: #ffcdd2;
        color: #d32f2f;
        padding: 3px 8px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
    }
    
    .recommendation {
        background: #f3e5f5;
        border-left: 5px solid #6a1b9a;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #4a148c;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #ccc, transparent);
        margin: 30px 0;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===== DICTIONNAIRE MULTILINGUE =====
translations = {
    'fr': {
        'title': '💧 Irrigation Intelligente',
        'subtitle': 'Tableau de Bord de Gestion des Ressources en Eau',
        'language': '🌍 Langue',
        'zone_selection': 'Sélectionner une Zone',
        'all_zones': 'Toutes les Zones',
        'soil_humidity': 'Humidité du Sol',
        'temperature': 'Température',
        'irrigation_time': 'Temps d\'Irrigation',
        'water_consumption': 'Consommation d\'Eau',
        'zone_details': 'Détails des Zones',
        'status': 'État',
        'recommendations': 'Recommandations',
        'optimal_irrigation': '✅ Irrigation Optimale',
        'optimal_text': 'Humidité entre 50-70%\nTemps idéal: Matin ou Soir',
        'warning': '⚠️ Attention Requise',
        'warning_text': 'Humidité < 50%\nRi dans les 2 heures',
        'critical': '🔴 Situation Critique',
        'critical_text': 'Humidité < 30%\nArrosage d\'urgence immédiate!',
        'charts': '📊 Analyses Visuelles',
        'footer': '🌱 Système de Gestion Intelligente du Riz - Mise à jour: ',
        'realtime_prediction': '🤖 Prédiction en Temps Réel',
        'select_zone_predict': 'Sélectionnez la zone pour la prédiction',
        'enter_sensor_data': 'Entrez les données des capteurs',
        'predict_button': '🔮 Prédire le temps d\'irrigation',
        'prediction_result': '📊 Résultat de la Prédiction',
        'predicted_time': 'Temps d\'irrigation prédit',
        'urgency_level': 'Niveau d\'urgence',
        'action_required': 'Action recommandée',
        'model_accuracy': 'Précision du modèle (R²)',
        'no_model': '⚠️ Modèle non disponible pour cette zone',
        'enter_data_first': 'Veuillez entrer les données des capteurs ci-dessus et cliquer sur Prédire',
        'minutes': 'minutes',
        'urgent_action': 'Irrigation immédiate requise',
        'high_action': 'Planifier irrigation dans les 2h',
        'medium_action': 'Irrigation dans la journée',
        'low_action': 'Conditions optimales, pas d\'urgence',
    },
    'ar': {
        'title': '💧 الري الذكي',
        'subtitle': 'لوحة تحكم إدارة موارد المياه',
        'language': '🌍 اللغة',
        'zone_selection': 'اختر المنطقة',
        'all_zones': 'جميع المناطق',
        'soil_humidity': 'رطوبة التربة',
        'temperature': 'درجة الحرارة',
        'irrigation_time': 'وقت الري',
        'water_consumption': 'استهلاك المياه',
        'zone_details': 'تفاصيل المناطق',
        'status': 'الحالة',
        'recommendations': 'التوصيات',
        'optimal_irrigation': '✅ الري الأمثل',
        'optimal_text': 'الرطوبة بين 50-70%\nالوقت المثالي: صباحا أو مساء',
        'warning': '⚠️ تحذير',
        'warning_text': 'الرطوبة أقل من 50%\nالري خلال ساعتين',
        'critical': '🔴 حالة حرجة',
        'critical_text': 'الرطوبة أقل من 30%\nري طوارئ فوري!',
        'charts': '📊 التحليلات البصرية',
        'footer': '🌱 نظام إدارة الري الذكي - آخر تحديث: ',
        'realtime_prediction': '🤖 التنبؤ في الوقت الفعلي',
        'select_zone_predict': 'اختر المنطقة للتنبؤ',
        'enter_sensor_data': 'أدخل بيانات أجهزة الاستشعار',
        'predict_button': '🔮 التنبؤ بوقت الري',
        'prediction_result': '📊 نتيجة التنبؤ',
        'predicted_time': 'وقت الري المتوقع',
        'urgency_level': 'مستوى الإلحاح',
        'action_required': 'الإجراء الموصى به',
        'model_accuracy': 'دقة النموذج (R²)',
        'no_model': '⚠️ النموذج غير متوفر لهذه المنطقة',
        'enter_data_first': 'يرجى إدخال بيانات أجهزة الاستشعار أعلاه والنقر على التنبؤ',
        'minutes': 'دقائق',
        'urgent_action': 'الري الفوري مطلوب',
        'high_action': 'خطط للري خلال ساعتين',
        'medium_action': 'الري خلال اليوم',
        'low_action': 'ظروف مثالية، لا استعجال',
    }
}

# ===== FONCTION DE CHARGEMENT DU MODÈLE =====
@st.cache_resource
def load_model(zone):
    """Charge le modèle ML pour une zone spécifique depuis le dossier models/"""
    try:
        # Chercher dans le dossier models/ avec le format edge_model_
        zone_lower = zone.lower()
        
        # Liste des chemins possibles (pour gérer l'encodage de Boghé)
        possible_paths = [
            f'models/edge_model_{zone_lower}.pkl',
            f'models/model_{zone_lower}.pkl',
            f'edge_model_{zone_lower}.pkl',
            f'model_{zone_lower}.pkl'
        ]
        
        # Cas spécial pour Boghé avec problème d'encodage
        if zone == 'Boghé':
            possible_paths.extend([
                'models/edge_model_bogh#U00e9.pkl',
                'models/edge_model_boghe.pkl',
                'edge_model_bogh#U00e9.pkl'
            ])
        
        # Chercher le fichier
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                return model_data
        
        # Si aucun fichier trouvé
        return None
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# ===== CHARGEMENT DES DONNÉES =====
@st.cache_data
def load_data():
    """Charge les données depuis le fichier CSV du projet"""
    csv_path = 'irrigation_dataset_mauritania.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error(f"❌ Fichier de données non trouvé: {csv_path}")
        # Générer des données d'exemple
        return generate_sample_data()

def generate_sample_data():
    """Génère des données d'exemple pour la démo"""
    np.random.seed(42)
    zones = ['Rosso', 'Kaedi', 'Boghé']
    data = []
    
    for zone in zones:
        for _ in range(50):
            humidity = np.random.uniform(20, 85)
            temperature = np.random.uniform(25, 40)
            ph = np.random.uniform(5.5, 7.5)
            evapotranspiration = np.random.uniform(3, 8)
            
            # Calcul du temps d'irrigation
            if humidity < 30:
                irrigation_time = np.random.uniform(15, 25)
            elif humidity < 50:
                irrigation_time = np.random.uniform(10, 15)
            elif humidity < 70:
                irrigation_time = np.random.uniform(5, 10)
            else:
                irrigation_time = np.random.uniform(2, 5)
            
            data.append({
                'zone': zone,
                'humidity': humidity,
                'temperature': temperature,
                'ph': ph,
                'evapotranspiration': evapotranspiration,
                'recommended_irrigation_time_min': irrigation_time
            })
    
    return pd.DataFrame(data)

# Chargement des données
df = load_data()

# ===== INITIALISATION DE L'ÉTAT DE SESSION =====
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.prediction_value = None
    st.session_state.urgence = None
    st.session_state.action = None

# ===== INTERFACE PRINCIPALE =====

# Sélection de la langue dans la barre latérale
with st.sidebar:
    lang_code = st.selectbox(
        '🌍 Language / اللغة',
        options=['fr', 'ar'],
        format_func=lambda x: 'Français 🇫🇷' if x == 'fr' else 'العربية 🇸🇦',
        key='language_selector'
    )
    
    st.markdown("---")
    
    # Sélection de zone pour filtrage
    selected_zone = st.selectbox(
        translations[lang_code]['zone_selection'],
        options=[translations[lang_code]['all_zones']] + list(df['zone'].unique()),
        key='zone_filter'
    )

t = translations[lang_code]

# Appliquer le style RTL pour l'arabe
if lang_code == 'ar':
    st.markdown('<div dir="rtl">', unsafe_allow_html=True)

# TITRE ET SOUS-TITRE
st.markdown(f'<h1 class="header-title">{t["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="header-subtitle">{t["subtitle"]}</p>', unsafe_allow_html=True)

# Filtrer les données selon la zone sélectionnée
if selected_zone == t['all_zones']:
    df_filtered = df
else:
    df_filtered = df[df['zone'] == selected_zone]

# ===== STATISTIQUES PRINCIPALES =====
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_humidity = df_filtered['humidity'].mean()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">💧 {t['soil_humidity']}</div>
        <div class="stat-value">{avg_humidity:.1f}<span class="stat-unit">%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_temp = df_filtered['temperature'].mean()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">🌡️ {t['temperature']}</div>
        <div class="stat-value">{avg_temp:.1f}<span class="stat-unit">°C</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_irrigation = df_filtered['recommended_irrigation_time_min'].mean()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">⏱️ {t['irrigation_time']}</div>
        <div class="stat-value">{avg_irrigation:.1f}<span class="stat-unit">min</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_water = (df_filtered['recommended_irrigation_time_min'] * 10).sum()
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">💦 {t['water_consumption']}</div>
        <div class="stat-value">{total_water:.0f}<span class="stat-unit">L</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== SECTION PRÉDICTION EN TEMPS RÉEL =====
st.markdown(f"### {t['realtime_prediction']}")

predict_col1, predict_col2 = st.columns([1, 1])

with predict_col1:
    st.markdown(f"#### {t['enter_sensor_data']}")
    
    # Sélection de zone pour la prédiction
    zone_for_prediction = st.selectbox(
        t['select_zone_predict'],
        options=df['zone'].unique(),
        key='prediction_zone_select'
    )
    
    # Sliders avec clés uniques basées sur la zone
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        humidity_input = st.slider(
            "💧 Humidité du Sol (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key=f'humidity_{zone_for_prediction}'
        )
        
        temperature_input = st.slider(
            "🌡️ Température (°C)",
            min_value=0,
            max_value=50,
            value=30,
            step=1,
            key=f'temperature_{zone_for_prediction}'
        )
    
    with col_input2:
        ph_input = st.slider(
            "⚗️ pH du Sol",
            min_value=4.0,
            max_value=9.0,
            value=6.5,
            step=0.1,
            key=f'ph_{zone_for_prediction}'
        )
        
        evapotranspiration_input = st.slider(
            "💨 Evapotranspiration (mm/jour)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            key=f'evapo_{zone_for_prediction}'
        )
    
    # Bouton de prédiction
    if st.button(t['predict_button'], type="primary", use_container_width=True, key='predict_btn'):
        # Charger le modèle pour la zone sélectionnée
        model_data = load_model(zone_for_prediction)
        
        if model_data is not None:
            # Préparer les données pour la prédiction
            sensor_data = pd.DataFrame([{
                'humidity': humidity_input,
                'temperature': temperature_input,
                'ph': ph_input,
                'evapotranspiration': evapotranspiration_input
            }])
            
            # Faire la prédiction
            prediction = model_data['model'].predict(sensor_data)[0]
            
            # Déterminer le niveau d'urgence (LOGIQUE CORRIGÉE)
            # Plus l'humidité est basse, plus le temps d'irrigation est long
            if prediction < 5:
                urgence = "🟢 FAIBLE"
                action = t['low_action']
            elif prediction < 10:
                urgence = "🟡 MOYENNE"
                action = t['medium_action']
            elif prediction < 15:
                urgence = "🟠 ÉLEVÉE"
                action = t['high_action']
            else:
                urgence = "🔴 CRITIQUE"
                action = t['urgent_action']
            
            # Stocker dans session_state
            st.session_state.prediction_made = True
            st.session_state.prediction_value = prediction
            st.session_state.urgence = urgence
            st.session_state.action = action
            st.session_state.model_r2 = model_data['metrics']['r2']
            st.session_state.zone_predicted = zone_for_prediction
        else:
            st.session_state.prediction_made = False
            st.session_state.model_error = True

with predict_col2:
    st.markdown(f"#### {t['prediction_result']}")
    
    # Afficher le résultat si une prédiction a été faite
    if st.session_state.prediction_made:
        # Zone prédite
        st.success(f"🌾 Zone: **{st.session_state.zone_predicted}**")
        
        # Temps d'irrigation prédit avec métrique
        st.metric(
            label=t['predicted_time'],
            value=f"{st.session_state.prediction_value:.1f}",
            delta=None,
            help="Temps calculé par le modèle ML"
        )
        st.caption(t['minutes'])
        
        # Niveau d'urgence avec couleur appropriée
        st.markdown(f"**{t['urgency_level']}**")
        if "CRITIQUE" in st.session_state.urgence:
            st.error(st.session_state.urgence)
        elif "ÉLEVÉE" in st.session_state.urgence:
            st.warning(st.session_state.urgence)
        elif "MOYENNE" in st.session_state.urgence:
            st.info(st.session_state.urgence)
        else:
            st.success(st.session_state.urgence)
        
        # Action recommandée
        st.markdown(f"**{t['action_required']}**")
        st.write(st.session_state.action)
        
        # Précision du modèle
        st.divider()
        st.caption(f"{t['model_accuracy']}: {st.session_state.model_r2:.3f}")
        
    elif hasattr(st.session_state, 'model_error') and st.session_state.model_error:
        st.error(t['no_model'])
        st.info("💡 Exécutez d'abord: `python edge_model.py`")
    else:
        st.info(t['enter_data_first'])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== DÉTAILS PAR ZONE =====
st.markdown(f"### {t['zone_details']}")

zones = df['zone'].unique()
zone_cols = st.columns(len(zones))

for idx, zone in enumerate(zones):
    with zone_cols[idx]:
        zone_data = df[df['zone'] == zone]
        humidity = zone_data['humidity'].mean()
        temp = zone_data['temperature'].mean()
        irrigation = zone_data['recommended_irrigation_time_min'].mean()
        
        # Déterminer le statut
        if humidity > 70:
            status = f'<span class="status-ok">✅ {t["optimal_irrigation"].split()[0]}</span>'
        elif humidity > 50:
            status = f'<span class="status-warning">⚠️ {t["warning"].split()[0]}</span>'
        else:
            status = f'<span class="status-critical">🔴 CRITIQUE</span>'
        
        st.markdown(f"""
        <div class="zone-card">
            <h3>🌾 {zone}</h3>
            <div class="zone-stat">
                <span class="zone-label">💧 Humidité:</span>
                <span class="zone-value">{humidity:.1f}%</span>
            </div>
            <div class="zone-stat">
                <span class="zone-label">🌡️ Température:</span>
                <span class="zone-value">{temp:.1f}°C</span>
            </div>
            <div class="zone-stat">
                <span class="zone-label">⏱️ Riz (min):</span>
                <span class="zone-value">{irrigation:.1f}</span>
            </div>
            <div class="zone-stat">
                <span class="zone-label">{t['status']}:</span>
                <span>{status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== GRAPHIQUES =====
st.markdown(f"### {t['charts']}")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_humidity = px.box(
        df_filtered,
        x='zone',
        y='humidity',
        color='zone',
        title=f"📊 {t['soil_humidity']} par zone",
        labels={'humidity': 'Humidité (%)', 'zone': 'Zone'},
        color_discrete_map={'Rosso': '#2e7d32', 'Kaedi': '#1565c0', 'Boghé': '#ff6f00'}
    )
    fig_humidity.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_humidity, use_container_width=True, key='chart_humidity')

with col_chart2:
    fig_temp = px.box(
        df_filtered,
        x='zone',
        y='temperature',
        color='zone',
        title=f"📊 {t['temperature']} par zone",
        labels={'temperature': 'Température (°C)', 'zone': 'Zone'},
        color_discrete_map={'Rosso': '#2e7d32', 'Kaedi': '#1565c0', 'Boghé': '#ff6f00'}
    )
    fig_temp.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_temp, use_container_width=True, key='chart_temp')

col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    irrigation_data = df_filtered.groupby('zone')['recommended_irrigation_time_min'].mean().reset_index()
    fig_irrigation = px.bar(
        irrigation_data,
        x='zone',
        y='recommended_irrigation_time_min',
        color='zone',
        title=f"📊 {t['irrigation_time']} recommandé",
        labels={'recommended_irrigation_time_min': 'Temps (min)', 'zone': 'Zone'},
        color_discrete_map={'Rosso': '#2e7d32', 'Kaedi': '#1565c0', 'Boghé': '#ff6f00'},
        text_auto='.1f'
    )
    fig_irrigation.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_irrigation, use_container_width=True, key='chart_irrigation')

with col_chart4:
    fig_scatter = px.scatter(
        df_filtered,
        x='humidity',
        y='recommended_irrigation_time_min',
        color='zone',
        title='Corrélation: Humidité vs Irrigation',
        labels={'humidity': 'Humidité (%)', 'recommended_irrigation_time_min': 'Temps (min)'},
        color_discrete_map={'Rosso': '#2e7d32', 'Kaedi': '#1565c0', 'Boghé': '#ff6f00'}
    )
    fig_scatter.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_scatter, use_container_width=True, key='chart_scatter')

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== RECOMMANDATIONS =====
st.markdown(f"### 💡 {t['recommendations']}")

rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    st.markdown(f"""
    <div class="recommendation" style="background: #c8e6c9; border-left-color: #2e7d32; color: #1b5e20;">
        <h4 style="color: #1b5e20;">{t['optimal_irrigation']}</h4>
        <p>{t['optimal_text']}</p>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    st.markdown(f"""
    <div class="recommendation" style="background: #fff9c4; border-left-color: #f57f17; color: #f57f17;">
        <h4 style="color: #f57f17;">{t['warning']}</h4>
        <p>{t['warning_text']}</p>
    </div>
    """, unsafe_allow_html=True)

with rec_col3:
    st.markdown(f"""
    <div class="recommendation" style="background: #ffcdd2; border-left-color: #d32f2f; color: #d32f2f;">
        <h4 style="color: #d32f2f;">{t['critical']}</h4>
        <p>{t['critical_text']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown(f"""
<div style='text-align: center; color: #999; padding: 20px; font-size: 0.9em;'>
    {t['footer']}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)

if lang_code == 'ar':
    st.markdown('</div>', unsafe_allow_html=True)