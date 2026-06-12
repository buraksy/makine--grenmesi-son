"""
🎵 Spotify Popularity Classification Dashboard — TASLAK
=======================================================
Sınıflandırma yönünde taslak; resmi uyarlanacak dosya app.py olacak.
Güncel metrikler ve model: README.md · models/best_classifier_rf.pkl

Model: RandomForest Classifier (Binary Classification)
- Popularity < 50 -> Düşük Popüler (0)
- Popularity >= 50 -> Yüksek Popüler (1)

Author: ML Team
Last Updated: 2026-06-10
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎵 Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(29, 185, 84, 0.3);
        color: white;
    }
    
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .hero p {
        font-size: 1.1rem;
        margin: 10px 0 0 0;
        opacity: 0.95;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(29, 185, 84, 0.5);
        box-shadow: 0 8px 32px rgba(29, 185, 84, 0.15);
    }
    
    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.05));
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .result-medium {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 152, 0, 0.05));
        border: 2px solid #FF9800;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .result-low {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(244, 67, 54, 0.05));
        border: 2px solid #F44336;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 0.95rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Metrics row */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 12px 36px rgba(29, 185, 84, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sliders */
    .stSlider {
        margin: 20px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(48, 43, 99, 0.8), rgba(36, 36, 62, 0.8));
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1DB954, #1ed760);
        border-radius: 8px;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {}

# ============================================================================
# LOAD MODEL AND PREPROCESSING PIPELINE
# ============================================================================

# ============================================================================
# LOAD MODEL AND PREPROCESSING PIPELINE
# ============================================================================

# Genre list'ini tanımla
GENRES = [
    'acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'anime',
    'black-metal', 'bluegrass', 'blues', 'brazil', 'breakbeat', 'british',
    'cantopop', 'chicago-house', 'children', 'chill', 'classical', 'club',
    'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house',
    'detroit-techno', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep',
    'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk',
    'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge',
    'guitar', 'happy', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal',
    'hip-hop', 'honky-tonk', 'house', 'idm', 'indian', 'indie', 'indie-pop',
    'industrial', 'iranian', 'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz',
    'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal',
    'metalcore', 'minimal-techno', 'mpb', 'new-age', 'opera', 'pagode',
    'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house',
    'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'reggae', 'reggaeton', 'rock',
    'rock-n-roll', 'rockabilly', 'romance', 'sad', 'salsa', 'samba',
    'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep',
    'songwriter', 'soul', 'spanish', 'study', 'swedish', 'synth-pop', 'tango',
    'techno', 'trance', 'trip-hop', 'turkish', 'world-music'
]

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and preprocessing scaler"""
    try:
        base_path = Path(__file__).parent / "models"
        
        model_path = base_path / "final_model.pkl"
        scaler_path = base_path / "preprocessing_pipeline_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            st.error("❌ Model files not found!")
            return None, None
        
        model = joblib.load(str(model_path))
        scaler = joblib.load(str(scaler_path))
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


def create_features(acousticness, danceability, energy, instrumentalness, 
                   liveness, loudness, mode, speechiness, tempo, time_signature,
                   duration_ms, key, explicit, selected_genres):
    """
    Create feature vector matching the training data preprocessing
    """
    try:
        # Log transformations
        duration_ms_log = np.log1p(duration_ms)
        speechiness_log = np.log1p(speechiness * 1000)  # Scale before log
        instrumentalness_log = np.log1p(instrumentalness * 1000)
        liveness_log = np.log1p(liveness * 1000)
        
        # Interaction features
        energy_loudness_interaction = energy * loudness
        
        # Mood score (combination of valence and energy)
        mood_score = (0.6 * (acousticness * 0.5 + danceability * 1.5) + 
                     0.4 * energy)
        
        # Electronic score
        electronic_score = (1 - acousticness) * energy
        
        # Create feature vector
        features = {
            'explicit': int(explicit),
            'danceability': danceability,
            'energy': energy,
            'key': key,
            'loudness': loudness,
            'mode': mode,
            'acousticness': acousticness,
            'valence': 0.5,  # Default middle value
            'tempo': tempo,
            'time_signature': time_signature,
            'duration_ms_log': duration_ms_log,
            'speechiness_log': speechiness_log,
            'instrumentalness_log': instrumentalness_log,
            'liveness_log': liveness_log,
            'energy_loudness_interaction': energy_loudness_interaction,
            'mood_score': mood_score,
            'electronic_score': electronic_score,
        }
        
        # Add genre one-hot encoding
        for genre in GENRES:
            features[f'genre_{genre}'] = 1 if genre in selected_genres else 0
        
        return pd.DataFrame([features])
    
    except Exception as e:
        st.error(f"❌ Error creating features: {str(e)}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_popularity_category(prediction_proba):
    """Categorize prediction probability"""
    if prediction_proba >= 0.7:
        return "🔥 Very Popular", "result-high", prediction_proba
    elif prediction_proba >= 0.5:
        return "📈 Popular", "result-high", prediction_proba
    elif prediction_proba >= 0.3:
        return "⚡ Moderate", "result-medium", prediction_proba
    else:
        return "📊 Not Popular", "result-low", prediction_proba

def create_gauge_chart(value, max_value=100, title="Popularity Score"):
    """Create a gauge chart for visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "#1DB954"},
            'steps': [
                {'range': [0, max_value * 0.3], 'color': "rgba(244, 67, 54, 0.2)"},
                {'range': [max_value * 0.3, max_value * 0.6], 'color': "rgba(255, 152, 0, 0.2)"},
                {'range': [max_value * 0.6, max_value], 'color': "rgba(76, 175, 80, 0.2)"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.8
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='white', size=12),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown("""
    <div class="hero">
        <h1>🎵 Spotify Track Popularity Predictor</h1>
        <p>Predict how popular your Spotify track will be using advanced machine learning models trained on 114,000+ songs</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### 🎚️ Track Features")
    st.markdown("Configure your track's audio characteristics:")
    
    # Audio Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎼 Audio Features**")
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3, 0.01)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1, 0.01)
        
    with col2:
        st.markdown("**🎤 Advanced Features**")
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -5.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)
        mode = st.selectbox("Mode", [0, 1], format_func=lambda x: "Major" if x == 1 else "Minor")
    
    col1, col2 = st.columns(2)
    with col1:
        tempo = st.slider("Tempo (BPM)", 0, 240, 120, 1)
        time_signature = st.selectbox("Time Signature", [3, 4, 5])
    
    with col2:
        duration_ms = st.slider("Duration (sec)", 30, 600, 180, 10)
        duration_ms = duration_ms * 1000  # Convert to milliseconds
        key = st.selectbox("Key", list(range(12)))
    
    st.markdown("---")
    
    # Genre Selection
    st.markdown("**🎵 Genre Selection**")
    selected_genres = st.multiselect(
        "Select genres (affects prediction):",
        GENRES,
        default=['pop', 'hip-hop']
    )
    
    # Track Characteristics
    explicit = st.checkbox("🔞 Explicit Content", value=False)

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.warning("⚠️ Could not load the model. Please check the model files.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Feature Analysis", "ℹ️ About"])

with tab1:
    # Display current feature summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Acousticness</div>
            <div class="stat-number" style="color: #1DB954;">{acousticness:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Danceability</div>
            <div class="stat-number" style="color: #1DB954;">{danceability:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Energy</div>
            <div class="stat-number" style="color: #1DB954;">{energy:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">Loudness</div>
            <div class="stat-number" style="color: #1DB954;">{loudness:.1f}dB</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("---")
    
    col_pred, col_reset = st.columns([1, 1])
    
    with col_pred:
        predict_button = st.button("🎯 Predict Popularity", use_container_width=True)
    
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.prediction_made = False
            st.session_state.prediction_result = None
            st.rerun()
    
    # Make prediction
    if predict_button:
        try:
            # Create features
            features_df = create_features(
                acousticness, danceability, energy, instrumentalness,
                liveness, loudness, mode, speechiness, tempo, time_signature,
                duration_ms, key, explicit, selected_genres
            )
            
            if features_df is not None:
                # Scale features
                features_scaled = scaler.transform(features_df)
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                
                # Store results
                st.session_state.prediction_made = True
                st.session_state.prediction_result = {
                    'prediction': prediction,
                    'probability': prediction_proba[1] if prediction == 1 else prediction_proba[0],
                    'proba_not_popular': prediction_proba[0],
                    'proba_popular': prediction_proba[1]
                }
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Display prediction results
    if st.session_state.get('prediction_made', False) and st.session_state.get('prediction_result'):
        st.markdown("---")
        
        result = st.session_state.prediction_result
        pred_value = result['proba_popular']
        category, css_class, _ = get_popularity_category(pred_value)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎵 Prediction",
                "POPULAR" if result['prediction'] == 1 else "NOT POPULAR",
                delta=f"Confidence: {pred_value*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "🎯 Popular Probability",
                f"{result['proba_popular']*100:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "📉 Not Popular Probability", 
                f"{result['proba_not_popular']*100:.1f}%",
                delta=None
            )
        
        # Probability bar chart
        fig_prob = go.Figure(data=[
            go.Bar(
                x=['Not Popular', 'Popular'],
                y=[result['proba_not_popular'], result['proba_popular']],
                marker_color=['#F44336', '#4CAF50'],
                text=[f"{result['proba_not_popular']*100:.1f}%", 
                      f"{result['proba_popular']*100:.1f}%"],
                textposition='auto',
            )
        ])
        
        fig_prob.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color='white'),
            height=400,
            showlegend=False,
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)

with tab2:
    st.markdown("### 📊 Feature Analysis")
    
    # Create feature importance visualization
    features_dict = {
        'Acousticness': acousticness,
        'Danceability': danceability,
        'Energy': energy,
        'Instrumentalness': instrumentalness,
        'Liveness': liveness,
        'Speechiness': speechiness,
        'Mode': mode,
    }
    
    # Bar chart
    fig_bar = px.bar(
        x=list(features_dict.values()),
        y=list(features_dict.keys()),
        orientation='h',
        title="Audio Features Distribution",
        labels={'x': 'Value', 'y': 'Feature'},
        color=list(features_dict.values()),
        color_continuous_scale=['#F44336', '#FF9800', '#4CAF50'],
    )
    
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color='white'),
        height=400,
        showlegend=False,
        xaxis_title="Feature Value",
        yaxis_title="",
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Radar chart
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=list(features_dict.values()),
        theta=list(features_dict.keys()),
        fill='toself',
        name='Features',
        line_color='#1DB954',
        fillcolor='rgba(29, 185, 84, 0.3)',
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(255,255,255,0.1)",
                linecolor="rgba(255,255,255,0.2)",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='white'),
        height=500,
        title="Audio Features Radar",
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Feature statistics
    st.markdown("### 📈 Feature Statistics")
    
    stats_df = pd.DataFrame({
        'Feature': list(features_dict.keys()),
        'Value': list(features_dict.values()),
        'Category': ['High' if v > 0.7 else 'Medium' if v > 0.3 else 'Low' for v in features_dict.values()]
    })
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Model Information
        - **Model Type**: Ensemble Learning (Bagging)
        - **Training Data**: 114,000+ Spotify tracks
        - **Features**: 11 audio features
        - **Performance Metrics**:
          - R² Score: 0.5694
          - RMSE: 14.58
          - MAE: 10.67
        """)
    
    with col2:
        st.markdown("""
        #### 🎵 Audio Features
        - **Acousticness**: Confidence measure of whether a track is acoustic
        - **Danceability**: Suitability of a track for dancing
        - **Energy**: Intensity and activity of a track
        - **Instrumentalness**: Probability of no vocals
        - **Liveness**: Presence of audience in recording
        - **Speechiness**: Presence of spoken words
        - **Valence**: Musical positiveness
        - **Loudness**: Overall loudness in decibels
        - **Tempo**: Speed of track (BPM)
        """)
    
    st.markdown("---")
    st.markdown("""
    #### 🚀 How to Use
    1. Adjust audio features in the sidebar
    2. Click **Predict Popularity** button
    3. View results in the Predictions tab
    4. Analyze feature distribution in Analysis tab
    
    #### ⚡ Tips for Better Predictions
    - High danceability and energy often correlate with popularity
    - Balanced features generally yield moderate popularity
    - Explicit content is tracked separately
    - Track duration matters (3-4 minutes is typical)
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; opacity: 0.6; font-size: 0.9rem;">
    <p>🎵 Spotify Popularity Predictor | ML-Powered Music Analytics</p>
    <p>Built with Streamlit & Scikit-learn | Data source: Spotify API</p>
    </div>
""", unsafe_allow_html=True)
