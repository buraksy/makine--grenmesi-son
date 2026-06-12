"""
Spotify Popülerlik Sınıflandırma — Streamlit Dashboard
======================================================
İkili sınıflandırma: 0–49 Düşük Popüler | 50–100 Yüksek Popüler
Model: Random Forest Classifier (models/best_classifier_rf.pkl)
Alternatif hibrit demo: archive/app_hybrid.py
"""

import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).parent.resolve()
ROOT_DIR = APP_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "best_classifier_rf.pkl"
SCALER_PATH = ROOT_DIR / "models" / "preprocessing_pipeline_scaler.pkl"
FEATURE_NAMES_PATH = ROOT_DIR / "feature_names.json"

POPULARITY_THRESHOLD = 50
METRICS = {
    "accuracy": 0.853,
    "f1_weighted": 0.842,
    "roc_auc": 0.902,
}

st.set_page_config(
    page_title="Spotify Popülerlik Sınıflandırma",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .hero {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        padding: 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: 0.5rem 0 0; opacity: 0.9; }
    .result-high {
        border: 2px solid #1DB954; border-radius: 12px; padding: 1.5rem;
        background: rgba(29, 185, 84, 0.08);
    }
    .result-low {
        border: 2px solid #e74c3c; border-radius: 12px; padding: 1.5rem;
        background: rgba(231, 76, 60, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        return None, None, None, []
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    if FEATURE_NAMES_PATH.exists():
        feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    else:
        feature_names = pd.read_csv(
            ROOT_DIR / "data" / "model_ready" / "X_train_clean.csv", nrows=0
        ).columns.tolist()
    genres = [c.replace("genre_", "") for c in feature_names if c.startswith("genre_")]
    return model, scaler, feature_names, genres


def build_feature_row(
    feature_names,
    *,
    explicit,
    danceability,
    energy,
    key,
    loudness,
    mode,
    acousticness,
    valence,
    tempo,
    time_signature,
    duration_ms,
    speechiness,
    instrumentalness,
    liveness,
    track_genre,
):
    """DataPrep pipeline ile uyumlu özellik vektörü."""
    row = {name: 0.0 for name in feature_names}

    row["explicit"] = int(explicit)
    row["danceability"] = danceability
    row["energy"] = energy
    row["key"] = key
    row["loudness"] = loudness
    row["mode"] = mode
    row["acousticness"] = acousticness
    row["valence"] = valence
    row["tempo"] = tempo
    row["time_signature"] = time_signature

    safe_duration_ms = max(int(duration_ms), 1)
    row["duration_ms_log"] = np.log1p(safe_duration_ms)
    row["speechiness_log"] = np.log1p(max(float(speechiness), 0.0))
    row["instrumentalness_log"] = np.log1p(max(float(instrumentalness), 0.0))
    row["liveness_log"] = np.log1p(max(float(liveness), 0.0))
    row["energy_loudness_interaction"] = energy * loudness
    row["mood_score"] = valence * energy
    row["electronic_score"] = 1.0 - acousticness

    genre_col = f"genre_{track_genre}"
    if genre_col in row:
        row[genre_col] = 1.0

    return pd.DataFrame([row], columns=feature_names)


def predict_track(model, scaler, features_df):
    if scaler is not None:
        X = pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns,
        )
    else:
        X = features_df
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return pred, float(proba[0]), float(proba[1])


FORM_DEFAULTS = {
    "danceability": 0.65,
    "energy": 0.7,
    "acousticness": 0.1,
    "valence": 0.5,
    "speechiness": 0.05,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "loudness": -5.0,
    "tempo": 120.0,
    "duration_sec": 200,
    "key": 5,
    "mode": 1,
    "time_signature": 4,
    "explicit": False,
    "track_genre": "pop",
}


def init_form_state(genres):
    for key, value in FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("track_genre") not in genres:
        st.session_state.track_genre = genres[0] if genres else "pop"
    if "prediction_done" not in st.session_state:
        st.session_state.prediction_done = False


def apply_random_features(genres):
    st.session_state.prediction_done = False
    st.session_state.danceability = round(random.uniform(0.15, 0.95), 2)
    st.session_state.energy = round(random.uniform(0.1, 0.98), 2)
    st.session_state.acousticness = round(random.uniform(0.0, 0.95), 2)
    st.session_state.valence = round(random.uniform(0.05, 0.95), 2)
    st.session_state.speechiness = round(random.uniform(0.0, 0.45), 2)
    st.session_state.instrumentalness = round(random.uniform(0.0, 0.85), 2)
    st.session_state.liveness = round(random.uniform(0.0, 0.8), 2)
    st.session_state.loudness = round(random.uniform(-25.0, -3.0), 1)
    st.session_state.tempo = round(random.uniform(70.0, 180.0), 0)
    st.session_state.duration_sec = random.randint(90, 360)
    st.session_state.key = random.randint(0, 11)
    st.session_state.mode = random.choice([0, 1])
    st.session_state.time_signature = random.choice([3, 4, 5])
    st.session_state.explicit = random.choice([True, False])
    st.session_state.track_genre = random.choice(genres) if genres else "pop"


model, scaler, feature_names, genres = load_assets()

with st.sidebar:
    st.markdown("### 🎵 Model")
    if model is not None:
        st.success("Random Forest Classifier")
        st.caption(f"Accuracy: {METRICS['accuracy']:.1%}")
        st.caption("131 özellik · notebook ile aynı model")
    else:
        st.error("Model yüklenemedi")

    st.markdown("---")
    page = st.radio(
        "Sayfalar",
        ["🏠 Ana Sayfa", "🎯 Tahmin", "ℹ️ Model Bilgisi"],
        label_visibility="collapsed",
    )

if page == "🏠 Ana Sayfa":
    st.markdown(
        """
        <div class="hero">
            <h1>🎵 Spotify Popülerlik Sınıflandırma</h1>
            <p>Şarkınızın <strong>Düşük</strong> (0–49) mi yoksa <strong>Yüksek</strong> (50–100) popülerlik
            sınıfına mı gireceğini tahmin edin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{METRICS['accuracy']:.1%}")
    c2.metric("F1 (weighted)", f"{METRICS['f1_weighted']:.1%}")
    c3.metric("ROC-AUC", f"{METRICS['roc_auc']:.2f}")
    c4.metric("Özellik", "131")

    st.markdown("### Sınıf Tanımları")
    st.markdown(
        """
        | Sınıf | Popularity aralığı | Anlam |
        |-------|-------------------|-------|
        | **0 — Düşük Popüler** | 0 – 49 | Düşük dinlenme / playlist görünürlüğü |
        | **1 — Yüksek Popüler** | 50 – 100 | Yüksek popülerlik |
        """
    )
    st.info(
        "Tahmin sayfasında özellikleri ayarlayıp **Sınıfı Tahmin Et** butonuna basarak sonucu alabilirsiniz."
    )

elif page == "🎯 Tahmin":
    st.markdown("## 🎯 Şarkı Popülerlik Tahmini")

    if model is None or not feature_names:
        st.error(
            f"`{MODEL_PATH.name}` bulunamadı. Önce notebook'u çalıştırıp modeli kaydedin "
            "veya `models/best_classifier_rf.pkl` dosyasının mevcut olduğundan emin olun."
        )
        st.stop()

    init_form_state(genres)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### Audio özellikleri")

        btn_rand, btn_reset = st.columns(2)
        with btn_rand:
            if st.button("🎲 Rastgele Doldur", use_container_width=True):
                apply_random_features(genres)
                st.rerun()
        with btn_reset:
            if st.button("↺ Varsayılan", use_container_width=True):
                st.session_state.prediction_done = False
                for k, v in FORM_DEFAULTS.items():
                    st.session_state[k] = v
                if st.session_state.track_genre not in genres:
                    st.session_state.track_genre = genres[0] if genres else "pop"
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            danceability = st.slider("Danceability", 0.0, 1.0, step=0.01, key="danceability")
            energy = st.slider("Energy", 0.0, 1.0, step=0.01, key="energy")
            acousticness = st.slider("Acousticness", 0.0, 1.0, step=0.01, key="acousticness")
            valence = st.slider("Valence", 0.0, 1.0, step=0.01, key="valence")
            speechiness = st.slider("Speechiness", 0.0, 1.0, step=0.01, key="speechiness")
        with c2:
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, step=0.01, key="instrumentalness")
            liveness = st.slider("Liveness", 0.0, 1.0, step=0.01, key="liveness")
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, step=0.5, key="loudness")
            tempo = st.slider("Tempo (BPM)", 0.0, 250.0, step=1.0, key="tempo")

        c3, c4 = st.columns(2)
        with c3:
            duration_sec = st.number_input("Süre (saniye)", 30, 900, key="duration_sec")
            key = st.selectbox("Key", list(range(12)), key="key")
        with c4:
            mode = st.selectbox(
                "Mode", [0, 1], format_func=lambda x: "Major" if x == 1 else "Minor", key="mode"
            )
            time_signature = st.selectbox("Time Signature", [3, 4, 5], key="time_signature")
            explicit = st.checkbox("Explicit içerik", key="explicit")

        track_genre = st.selectbox("Tür (Genre)", genres, key="track_genre")

        form_values = dict(
            explicit=explicit,
            danceability=danceability,
            energy=energy,
            key=key,
            loudness=loudness,
            mode=mode,
            acousticness=acousticness,
            valence=valence,
            tempo=tempo,
            time_signature=time_signature,
            duration_ms=int(duration_sec * 1000),
            speechiness=speechiness,
            instrumentalness=instrumentalness,
            liveness=liveness,
            track_genre=track_genre,
        )

        if st.button("🎯 Sınıfı Tahmin Et", type="primary", use_container_width=True):
            st.session_state.prediction_done = True
            st.session_state.prediction_input = form_values

    with col_result:
        st.markdown("#### Sonuç")

        if not st.session_state.prediction_done or "prediction_input" not in st.session_state:
            st.caption("Özellikleri ayarlayıp **Sınıfı Tahmin Et** butonuna basın.")
            st.stop()

        form_values = st.session_state.prediction_input
        features_df = build_feature_row(feature_names, **form_values)
        pred, p_low, p_high = predict_track(model, scaler, features_df)

        label = "Yüksek Popüler" if pred == 1 else "Düşük Popüler"
        css = "result-high" if pred == 1 else "result-low"
        icon = "🔥" if pred == 1 else "📉"
        confidence = p_high if pred == 1 else p_low

        st.markdown(
            f"""
            <div class="{css}">
                <h2>{icon} {label}</h2>
                <p>Sınıf: <strong>{pred}</strong> · Güven: <strong>{confidence:.1%}</strong></p>
                <p>Eşik: popularity &lt; {POPULARITY_THRESHOLD} → Düşük</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Düşük (0)", "Yüksek (1)"],
                    y=[p_low, p_high],
                    marker_color=["#e74c3c", "#1DB954"],
                    text=[f"{p_low:.1%}", f"{p_high:.1%}"],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title="Sınıf Olasılıkları",
            yaxis=dict(range=[0, 1.05], tickformat=".0%"),
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            top = imp.nlargest(10).sort_values()
            fig_imp = go.Figure(
                go.Bar(x=top.values, y=top.index, orientation="h", marker_color="#1DB954")
            )
            fig_imp.update_layout(title="Top 10 Özellik Önemi (Random Forest)", height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

        st.caption(
            "Bu uygulama notebook'taki **Random Forest** modelini kullanır. "
            "Tür (genre) tahmine en güçlü etkiyi yapar."
        )

else:
    st.markdown("## ℹ️ Model Bilgisi")
    st.markdown(
        f"""
        | Özellik | Değer |
        |---------|-------|
        | **Algoritma** | Random Forest Classifier |
        | **Problem** | İkili sınıflandırma |
        | **Eşik** | popularity &lt; {POPULARITY_THRESHOLD} → Düşük, ≥ {POPULARITY_THRESHOLD} → Yüksek |
        | **Eğitim verisi** | 114.000 şarkı (91.200 train) |
        | **Özellik sayısı** | 131 |
        | **class_weight** | balanced |
        | **Test Accuracy** | {METRICS['accuracy']:.3f} |
        | **Test F1 (weighted)** | {METRICS['f1_weighted']:.3f} |
        | **Test ROC-AUC** | {METRICS['roc_auc']:.3f} |
        | **Model dosyası** | `models/best_classifier_rf.pkl` |
        """
    )
    st.markdown("### Sınırlamalar")
    st.markdown(
        """
        - Audio özellikleri popülerliği zayıf açıklar; **tür (genre)** en güçlü sinyaldir.
        - Yüksek popüler sınıf azınlıkta (~%26); recall bu sınıfta daha düşük olabilir.
        - Sanatçı ünü, playlist sayısı gibi harici faktörler modele dahil değildir.
        """
    )
    st.markdown("### Alternatif demo")
    st.markdown(
        """
        Etkileşimli slider demo için (LR + tür hibrit): `archive/app_hybrid.py`

        ```bash
        streamlit run archive/app_hybrid.py
        ```
        """
    )
    st.markdown("### Kaynak")
    st.code("notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb", language="text")

st.markdown("---")
st.caption("Spotify Popülerlik Sınıflandırma · Random Forest · README.md")
