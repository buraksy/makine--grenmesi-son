"""
🎵 Spotify Popularity Dashboard — ARŞİV (Regresyon)
===================================================
⚠️ Eski regresyon prototipi. Güncel proje: ikili sınıflandırma.
Kaynak: README.md · Model: models/best_classifier_rf.pkl

Model: BaggingRegressor (Regression) — ARŞİV
- Output: Popularity score (0-100)

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

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎵 Spotify Popularity Classifier",
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
    .result-popular {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.05));
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
    }
    
    .result-not-popular {
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

# ============================================================================
# GENRE LIST AND CONSTANTS
# ============================================================================

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

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and preprocessing scaler"""
    try:
        base_path = Path(__file__).parent / "models"
        
        model_path = base_path / "best_model_clean.pkl"  # ← Düzeltildi
        scaler_path = base_path / "preprocessing_pipeline_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            st.error("❌ Model files not found!")
            return None, None
        
        model = joblib.load(str(model_path))
        scaler = joblib.load(str(scaler_path))
        
        print(f"DEBUG: Model n_features_in_: {model.n_features_in_}")
        print(f"DEBUG: Scaler n_features_in_: {scaler.n_features_in_}")
        
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
        speechiness_log = np.log1p(speechiness * 1000)
        instrumentalness_log = np.log1p(instrumentalness * 1000)
        liveness_log = np.log1p(liveness * 1000)
        
        # Interaction features
        energy_loudness_interaction = energy * loudness
        
        # Mood score
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
            'valence': 0.5,
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
        
        features_df = pd.DataFrame([features])
        print(f"DEBUG: Features shape: {features_df.shape}")
        print(f"DEBUG: Feature columns count: {len(features_df.columns)}")
        return features_df
    
    except Exception as e:
        st.error(f"❌ Error creating features: {str(e)}")
        return None

# ============================================================================
# HERO SECTION
# ============================================================================

st.markdown("""
    <div class="hero">
        <h1>🎵 Spotify Track Popularity Predictor</h1>
        <p>Predict your Spotify track's popularity score (0-100) using ML trained on 114,000+ songs</p>
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
        duration_sec = st.slider("Duration (sec)", 30, 600, 180, 10)
        duration_ms = duration_sec * 1000
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
    # Display feature summary
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
    
    # Prediction buttons
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
                
                # Make prediction (Regression - outputs popularity score)
                prediction_score = model.predict(features_scaled)[0]
                
                # Clamp to 0-100 range
                prediction_score = max(0, min(100, prediction_score))
                
                # Store results
                st.session_state.prediction_made = True
                st.session_state.prediction_result = {
                    'score': prediction_score,
                    'is_popular': prediction_score >= 50
                }
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
    
    # Display prediction results
    if st.session_state.get('prediction_made', False) and st.session_state.get('prediction_result'):
        st.markdown("---")
        
        result = st.session_state.prediction_result
        score = result['score']
        is_popular = result['is_popular']
        
        # Determine category
        if score >= 70:
            category = "🔥 Very Popular"
            color = "#4CAF50"
        elif score >= 50:
            category = "📈 Popular"
            color = "#4CAF50"
        elif score >= 30:
            category = "⚡ Moderate"
            color = "#FF9800"
        else:
            category = "📊 Not Popular"
            color = "#F44336"
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🎵 Popularity Score",
                f"{score:.1f}/100",
                delta=None
            )
        
        with col2:
            st.metric(
                "📊 Category",
                category,
                delta=None
            )
        
        with col3:
            st.metric(
                "🎯 Status", 
                "POPULAR ✓" if is_popular else "NOT POPULAR ✗",
                delta=None
            )
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Popularity Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(244, 67, 54, 0.2)"},
                    {'range': [30, 50], 'color': "rgba(255, 152, 0, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(76, 175, 80, 0.2)"},
                ]
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color='white', size=14),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.markdown("### 📊 Feature Analysis")
    
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
    
    # Genre distribution
    if selected_genres:
        st.markdown("### 🎵 Selected Genres")
        genre_cols = st.columns(len(selected_genres) if len(selected_genres) <= 5 else 5)
        for i, genre in enumerate(selected_genres[:5]):
            with genre_cols[i % 5]:
                st.info(f"✓ {genre}")
        if len(selected_genres) > 5:
            st.caption(f"...and {len(selected_genres) - 5} more")

with tab3:
    st.markdown("### ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Model Information
        - **Model Type**: Bagging Regressor
        - **Training Data**: 114,000+ Spotify tracks
        - **Training Features**: 131 engineered features
        - **Task**: Popularity Score Prediction (0-100)
        - **Output**: 
          - Popular: Score ≥ 50
          - Not Popular: Score < 50
        - **Features**:
          - Raw audio features
          - Log-transformed features
          - Interaction features
          - Genre one-hot encoding (113 genres)
        """)
    
    with col2:
        st.markdown("""
        #### 🎵 Audio Features Explained
        - **Acousticness**: 0.0-1.0, confidence of acoustic sound
        - **Danceability**: 0.0-1.0, suitability for dancing
        - **Energy**: 0.0-1.0, intensity and activity level
        - **Instrumentalness**: 0.0-1.0, absence of vocals
        - **Liveness**: 0.0-1.0, presence of live audience
        - **Loudness**: in dB, overall loudness
        - **Speechiness**: 0.0-1.0, spoken words presence
        - **Tempo**: BPM (beats per minute)
        - **Duration**: Track length in seconds
        """)
    
    st.markdown("---")
    st.markdown("""
    #### 🚀 How to Use
    1. **Configure Features**: Adjust audio characteristics in the sidebar
    2. **Select Genres**: Choose genres matching your track
    3. **Predict**: Click "Predict Popularity" for results
    4. **Analyze**: View feature importance in Analysis tab
    
    #### 💡 Tips for Better Predictions
    - **Popular tracks** typically have: high danceability, moderate-high energy
    - **Genre selection** significantly impacts predictions
    - **Explicit content** is tracked as a separate feature
    - **Typical durations**: 3-4 minutes correlate with higher popularity
    - **Loudness**: -5 to -3 dB is common for popular tracks
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; opacity: 0.6; font-size: 0.9rem;">
    <p>🎵 Spotify Popularity Predictor | ML-Powered Music Analytics</p>
    <p>Built with Streamlit & Scikit-learn | Bagging Regressor Model</p>
    </div>
""", unsafe_allow_html=True)
