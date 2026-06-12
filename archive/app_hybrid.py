"""
Spotify Popülerlik Sınıflandırma — Streamlit Dashboard
======================================================
İkili sınıflandırma: 0–49 Düşük Popüler | 50–100 Yüksek Popüler
Tahmin: Hibrit model (audio LR + tür önceliği) — slider'lar görünür etki eder.
Arşiv RF: models/best_classifier_rf.pkl (notebook eğitimi)
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

APP_DIR = Path(__file__).resolve().parent.parent
AUDIO_MODEL_PATH = APP_DIR / "models" / "audio_classifier_lr.pkl"
GENRE_PRIORS_PATH = APP_DIR / "models" / "genre_priors.json"
HYBRID_METRICS_PATH = APP_DIR / "models" / "hybrid_metrics.json"
SCALER_PATH = APP_DIR / "models" / "preprocessing_pipeline_scaler.pkl"
FEATURE_NAMES_PATH = APP_DIR / "feature_names.json"

POPULARITY_THRESHOLD = 50

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
    .metric-card {
        background: #1e1e1e; border: 1px solid #333; border-radius: 12px;
        padding: 1rem; text-align: center;
    }
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


def _load_metrics():
    defaults = {"accuracy": 0.774, "f1_weighted": 0.761, "roc_auc": 0.793}
    if HYBRID_METRICS_PATH.exists():
        data = json.loads(HYBRID_METRICS_PATH.read_text(encoding="utf-8"))
        return data.get("hybrid_interactive", defaults)
    return defaults


METRICS = _load_metrics()


@st.cache_resource
def load_assets():
    if not AUDIO_MODEL_PATH.exists() or not GENRE_PRIORS_PATH.exists():
        return None, None, None, None, []
    audio_model = joblib.load(AUDIO_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    genre_config = json.loads(GENRE_PRIORS_PATH.read_text(encoding="utf-8"))
    if FEATURE_NAMES_PATH.exists():
        feature_names = json.loads(FEATURE_NAMES_PATH.read_text(encoding="utf-8"))
    else:
        feature_names = pd.read_csv(
            APP_DIR / "data" / "model_ready" / "X_train_clean.csv", nrows=0
        ).columns.tolist()
    audio_cols = genre_config.get(
        "audio_feature_names",
        [c for c in feature_names if not c.startswith("genre_")],
    )
    genres = [c.replace("genre_", "") for c in feature_names if c.startswith("genre_")]
    return audio_model, scaler, feature_names, genre_config, audio_cols, genres


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


def scale_features(scaler, features_df):
    if scaler is not None:
        return pd.DataFrame(
            scaler.transform(features_df),
            columns=features_df.columns,
        )
    return features_df


def predict_hybrid(audio_model, scaler, features_df, genre_config, audio_cols):
    """Audio LR + tür önceliği ile hibrit olasılık."""
    X = scale_features(scaler, features_df)
    p_audio = float(audio_model.predict_proba(X[audio_cols])[0][1])

    genre = features_df.filter(regex=r"^genre_").idxmax(axis=1).iloc[0]
    genre_name = genre.replace("genre_", "") if genre.startswith("genre_") else "pop"
    priors = genre_config["priors"]
    p_genre = float(priors.get(genre_name, genre_config["global_prior"]))

    w = float(genre_config.get("audio_blend_weight", 0.48))
    p_high = float(np.clip(w * p_audio + (1.0 - w) * p_genre, 0.0, 1.0))
    p_low = 1.0 - p_high
    pred = int(p_high >= 0.5)
    return pred, p_low, p_high, p_audio, p_genre, w


def compute_slider_impacts(
    audio_model,
    scaler,
    feature_names,
    genre_config,
    audio_cols,
    form_values,
):
    """Her slider'ın P(yüksek) üzerindeki marjinal etkisini hesapla."""
    base_df = build_feature_row(feature_names, **form_values)
    _, _, p_high_base, _, _, _ = predict_hybrid(
        audio_model, scaler, base_df, genre_config, audio_cols
    )

    probes = [
        (
            "Süre (saniye)",
            "duration_ms",
            lambda v, d: int(np.clip(v + d * 1000, 30_000, 900_000)),
            60,
        ),
        ("Danceability", "danceability", lambda v, d: float(np.clip(v + d, 0, 1)), 0.15),
        ("Energy", "energy", lambda v, d: float(np.clip(v + d, 0, 1)), 0.15),
        ("Acousticness", "acousticness", lambda v, d: float(np.clip(v + d, 0, 1)), 0.15),
        ("Valence", "valence", lambda v, d: float(np.clip(v + d, 0, 1)), 0.15),
        ("Speechiness", "speechiness", lambda v, d: float(np.clip(v + d, 0, 1)), 0.1),
        ("Instrumentalness", "instrumentalness", lambda v, d: float(np.clip(v + d, 0, 1)), 0.15),
        ("Liveness", "liveness", lambda v, d: float(np.clip(v + d, 0, 1)), 0.1),
        ("Loudness (dB)", "loudness", lambda v, d: float(np.clip(v + d, -60, 0)), 5.0),
        ("Tempo (BPM)", "tempo", lambda v, d: float(np.clip(v + d, 0, 250)), 20.0),
    ]

    impacts = []
    for label, key, transform, delta in probes:
        if key == "duration_ms":
            current = form_values["duration_ms"]
        else:
            current = form_values[key]

        up_vals = dict(form_values)
        down_vals = dict(form_values)
        up_vals[key] = transform(current, delta)
        down_vals[key] = transform(current, -delta)

        up_df = build_feature_row(feature_names, **up_vals)
        down_df = build_feature_row(feature_names, **down_vals)
        _, _, p_up, _, _, _ = predict_hybrid(
            audio_model, scaler, up_df, genre_config, audio_cols
        )
        _, _, p_down, _, _, _ = predict_hybrid(
            audio_model, scaler, down_df, genre_config, audio_cols
        )
        spread = p_up - p_down
        impacts.append(
            {
                "Özellik": label,
                "Δ P(yüksek)": spread,
                "Yukarı": p_up - p_high_base,
                "Aşağı": p_down - p_high_base,
            }
        )

    return pd.DataFrame(impacts).sort_values("Δ P(yüksek)", key=abs, ascending=False)


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


audio_model, scaler, feature_names, genre_config, audio_cols, genres = load_assets()

with st.sidebar:
    st.markdown("### 🎵 Model")
    if audio_model is not None:
        st.success("Hibrit (Audio LR + Tür)")
        st.caption(f"Accuracy: {METRICS['accuracy']:.1%}")
        st.caption("Slider'lar bu modelde etkilidir")
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

    if audio_model is None or feature_names is None:
        st.error(
            "Hibrit model bulunamadı. `scripts/train_hybrid_classifier.py` çalıştırın "
            "veya `models/audio_classifier_lr.pkl` dosyasının mevcut olduğundan emin olun."
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
        track_genre = form_values["track_genre"]

        features_df = build_feature_row(feature_names, **form_values)
        pred, p_low, p_high, p_audio, p_genre, blend_w = predict_hybrid(
            audio_model, scaler, features_df, genre_config, audio_cols
        )

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

        st.caption(
            f"Audio skoru: **{p_audio:.1%}** · Tür önceliği ({track_genre}): **{p_genre:.1%}** · "
            f"Karışım: {blend_w:.0%} audio + {1-blend_w:.0%} tür"
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
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        impacts = compute_slider_impacts(
            audio_model, scaler, feature_names, genre_config, audio_cols, form_values
        )
        colors = ["#1DB954" if v >= 0 else "#e74c3c" for v in impacts["Δ P(yüksek)"]]
        fig_imp = go.Figure(
            go.Bar(
                x=impacts["Δ P(yüksek)"],
                y=impacts["Özellik"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1%}" for v in impacts["Δ P(yüksek)"]],
                textposition="outside",
            )
        )
        fig_imp.update_layout(
            title="Slider Etkisi (± değişimde P(yüksek) farkı)",
            xaxis_tickformat=".0%",
            height=420,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Özellikleri değiştirdikten sonra yeniden **Sınıfı Tahmin Et** butonuna basın. "
            "Alttaki grafik her slider'ın tahmine etkisini gösterir."
        )

else:
    st.markdown("## ℹ️ Model Bilgisi")
    st.markdown(
        f"""
        | Özellik | Değer |
        |---------|-------|
        | **Tahmin modeli (UI)** | Hibrit: Logistic Regression (audio) + tür önceliği |
        | **Notebook modeli** | Random Forest (131 özellik, arşiv) |
        | **Problem** | İkili sınıflandırma |
        | **Eşik** | popularity &lt; {POPULARITY_THRESHOLD} → Düşük, ≥ {POPULARITY_THRESHOLD} → Yüksek |
        | **Test Accuracy (hibrit)** | {METRICS['accuracy']:.3f} |
        | **Test F1 (hibrit)** | {METRICS['f1_weighted']:.3f} |
        | **Test ROC-AUC (hibrit)** | {METRICS['roc_auc']:.3f} |
        """
    )
    st.markdown("### Neden hibrit model?")
    st.markdown(
        """
        Eski Random Forest modeli **tür (genre)** özelliklerine aşırı bağımlıydı; süre ve tempo gibi
        slider'ları değiştirmek tahmini neredeyse hiç değiştirmiyordu. Hibrit modelde:

        - **Audio özellikleri** (süre, energy, valence vb.) doğrudan Logistic Regression ile skorlanır.
        - **Tür** eğitim verisindeki yüksek/düşük popülerlik oranından öncelik alır.
        - İkisi birleştirilerek hem slider'lar etkili olur hem tür bilgisi korunur.
        """
    )
    st.markdown("### Kaynak")
    st.code(
        "scripts/train_hybrid_classifier.py\nnotebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb",
        language="text",
    )

st.markdown("---")
st.caption("Spotify Popülerlik Sınıflandırma · Hibrit Audio+Tür · README.md")
