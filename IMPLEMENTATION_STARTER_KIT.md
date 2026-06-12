# 🚀 RECOMMENDATION SYSTEM - IMPLEMENTATION STARTER KIT

Quick-start code templates for immediate implementation.

> **Haziran 2026 güncellemesi:** Ana model **Random Forest Classifier** (ikili sınıf: Düşük 0–49 / Yüksek 50–100).  
> Model dosyası: `models/best_classifier_rf.pkl` · Pipeline: `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb`  
> Aşağıdaki SHAP/explainer örnekleri regresyon içindir — sınıflandırma için `TreeExplainer` veya `permutation_importance` kullanın.

---

## 📦 STEP 1: Install Additional Dependencies

```bash
pip install scipy>=1.11.0
pip install scikit-learn>=1.4.0
pip install shap>=0.44.0
pip install openai  # Optional, for LLM integration
```

---

## 🧩 STEP 2: Create Module Structure

```bash
# Create directories
mkdir -p modules
mkdir -p pages
mkdir -p config
mkdir -p data/reference

# Create __init__.py
touch modules/__init__.py
```

---

## 📝 STEP 3: Explainer Module (modules/explainer.py)

```python
"""
Explainer Module - WHY did the model predict this?
"""

import numpy as np
import pandas as pd
import shap
from pathlib import Path

class ExplainerModule:
    def __init__(self, model, background_data, feature_names):
        """
        Initialize explainer with model and background data.
        
        Args:
            model: Trained prediction model
            background_data: Sample of training data for SHAP
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        # Create SHAP explainer (cached for performance)
        print("Creating SHAP explainer...")
        self.explainer = shap.KernelExplainer(
            model.predict, 
            background_data.sample(100, random_state=42)  # Use 100 samples for speed
        )
        print("✅ SHAP explainer ready")
    
    def explain_prediction(self, input_features):
        """
        Generate comprehensive explanation for a prediction.
        
        Returns:
            dict with explanation components
        """
        # Get prediction
        if isinstance(input_features, pd.DataFrame):
            input_array = input_features.values
        else:
            input_array = input_features
        
        prediction = self.model.predict(input_array)[0]
        
        # Calculate SHAP values
        print("Computing SHAP values...")
        shap_values = self.explainer.shap_values(input_array)
        
        # Handle shape
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Get top features
        top_features = self._get_top_features(shap_values, input_array[0])
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation(
            prediction, top_features, input_features
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(shap_values)
        
        return {
            'prediction': prediction,
            'shap_values': shap_values,
            'top_features': top_features,
            'explanation_text': explanation_text,
            'confidence_score': confidence,
            'base_value': self.explainer.expected_value
        }
    
    def _get_top_features(self, shap_values, input_values, top_k=5):
        """Get top K most impactful features."""
        # Get absolute impacts
        impacts = [(self.feature_names[i], shap_values[i], input_values[i]) 
                   for i in range(len(shap_values))]
        
        # Sort by absolute impact
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return impacts[:top_k]
    
    def _generate_explanation(self, prediction, top_features, input_features):
        """Generate natural language explanation."""
        explanation = f"🎵 **Predicted Popularity: {prediction:.1f}/100**\n\n"
        explanation += "**Main Drivers:**\n\n"
        
        for feature_name, impact, value in top_features:
            if impact > 0:
                explanation += f"✅ **{feature_name}** ({value:.2f}): +{impact:.1f} points\n"
                explanation += f"   _{self._get_positive_reason(feature_name, value)}_\n\n"
            else:
                explanation += f"⚠️ **{feature_name}** ({value:.2f}): {impact:.1f} points\n"
                explanation += f"   _{self._get_negative_reason(feature_name, value)}_\n\n"
        
        return explanation
    
    def _get_positive_reason(self, feature_name, value):
        """Get reason for positive impact."""
        reasons = {
            'danceability': "High danceability attracts mainstream listeners",
            'energy': "Strong energy creates engagement",
            'valence': "Positive mood resonates with wider audiences",
            'acousticness': "Acoustic elements add authenticity",
            'track_genre': "This genre has strong popularity potential",
            'loudness': "Good loudness levels for streaming platforms",
            'tempo': "This tempo works well for the genre"
        }
        return reasons.get(feature_name, f"This {feature_name} value contributes positively")
    
    def _get_negative_reason(self, feature_name, value):
        """Get reason for negative impact."""
        reasons = {
            'speechiness': "High speechiness may limit musical appeal",
            'instrumentalness': "Lack of vocals can reduce mainstream appeal",
            'liveness': "Live recording quality may affect production value",
            'acousticness': "Low acoustic presence may lack warmth",
            'duration_ms': "Song length may not be optimal for streaming"
        }
        return reasons.get(feature_name, f"This {feature_name} value has negative impact")
    
    def _calculate_confidence(self, shap_values):
        """Calculate prediction confidence (0-100)."""
        # Higher variation in SHAP values = lower confidence
        variation = np.std(np.abs(shap_values))
        
        # Normalize to 0-100 scale (inverse relationship)
        confidence = max(0, min(100, 100 - (variation * 10)))
        
        return confidence


# Helper function to create explainer from saved model
def create_explainer(model_path, train_data_path):
    """
    Create explainer from saved model and training data.
    
    Usage:
        explainer = create_explainer(
            'models/best_model_clean.pkl',
            'data/model_ready/X_train_clean.csv'
        )
    """
    import joblib
    
    # Load model
    model = joblib.load(model_path)
    
    # Load training data
    train_data = pd.read_csv(train_data_path)
    feature_names = list(train_data.columns)
    
    # Create explainer
    explainer = ExplainerModule(model, train_data, feature_names)
    
    return explainer


# Test function
if __name__ == "__main__":
    # Test the explainer
    explainer = create_explainer(
        '../models/best_model_clean.pkl',
        '../data/model_ready/X_train_clean.csv'
    )
    
    # Load sample
    test_data = pd.read_csv('../data/model_ready/X_test_clean.csv')
    sample = test_data.iloc[0:1]
    
    # Explain
    result = explainer.explain_prediction(sample)
    
    print("\n" + "="*50)
    print(result['explanation_text'])
    print(f"\nConfidence: {result['confidence_score']:.1f}%")
    print("="*50)
```

---

## 🎼 STEP 4: Recommender Module (modules/recommender.py)

```python
"""
Recommender Module - HOW can I improve this song?
"""

import numpy as np
import pandas as pd

class RecommenderModule:
    def __init__(self, model, explainer, reference_data):
        """
        Initialize recommender with model and reference data.
        
        Args:
            model: Trained prediction model
            explainer: ExplainerModule instance
            reference_data: DataFrame of successful songs
        """
        self.model = model
        self.explainer = explainer
        self.reference_data = reference_data
        
        # Filter successful songs (popularity >= 60)
        self.successful_songs = reference_data[
            reference_data['popularity'] >= 60
        ] if 'popularity' in reference_data.columns else reference_data
        
        print(f"✅ Recommender initialized with {len(self.successful_songs)} successful songs")
    
    def generate_recommendations(self, input_features, current_prediction, 
                                 genre=None, max_recommendations=5):
        """
        Generate actionable recommendations to improve popularity.
        
        Returns:
            List of recommendation dicts
        """
        recommendations = []
        
        # Get SHAP explanation
        explanation = self.explainer.explain_prediction(input_features)
        top_features = explanation['top_features']
        
        # Identify features with negative impact
        negative_features = [
            (name, impact, value) 
            for name, impact, value in top_features 
            if impact < -1.0  # Only significant negative impacts
        ]
        
        # If no negative features, find underperforming features
        if len(negative_features) == 0:
            negative_features = self._find_underperforming_features(
                input_features, genre
            )
        
        for feature_name, current_impact, current_value in negative_features[:max_recommendations]:
            # Get optimal value for this feature
            optimal_value = self._get_optimal_value(feature_name, genre)
            
            # Estimate improvement
            estimated_gain = abs(current_impact) * 0.7  # Conservative estimate
            
            # Generate actionable advice
            advice = self._get_actionable_advice(
                feature_name, current_value, optimal_value
            )
            
            # Assess difficulty
            difficulty = self._assess_difficulty(feature_name, current_value, optimal_value)
            
            recommendations.append({
                'feature': feature_name,
                'current_value': round(current_value, 3),
                'suggested_value': round(optimal_value, 3),
                'estimated_gain': round(estimated_gain, 1),
                'current_impact': round(current_impact, 1),
                'actionable_advice': advice,
                'difficulty': difficulty,
                'priority': 'High' if estimated_gain > 5 else 'Medium'
            })
        
        # Sort by potential impact
        recommendations.sort(key=lambda x: x['estimated_gain'], reverse=True)
        
        return recommendations
    
    def _find_underperforming_features(self, input_features, genre):
        """Find features that are below optimal values."""
        underperforming = []
        
        audio_features = ['danceability', 'energy', 'valence', 
                         'acousticness', 'loudness', 'tempo']
        
        for feature in audio_features:
            if feature not in input_features.columns:
                continue
            
            current_value = input_features[feature].values[0]
            optimal_value = self._get_optimal_value(feature, genre)
            
            # Check if significantly below optimal
            if abs(current_value - optimal_value) > 0.1:
                gap = optimal_value - current_value
                underperforming.append((feature, -abs(gap) * 5, current_value))
        
        return sorted(underperforming, key=lambda x: x[1])[:5]
    
    def _get_optimal_value(self, feature_name, genre=None):
        """Get optimal feature value from successful songs."""
        if genre and 'track_genre' in self.successful_songs.columns:
            genre_songs = self.successful_songs[
                self.successful_songs['track_genre'] == genre
            ]
            if len(genre_songs) > 10:
                # Use genre-specific data
                return genre_songs[feature_name].quantile(0.75)
        
        # Fallback to all successful songs
        if feature_name in self.successful_songs.columns:
            return self.successful_songs[feature_name].quantile(0.75)
        else:
            # Default values if feature not found
            defaults = {
                'danceability': 0.65,
                'energy': 0.60,
                'valence': 0.55,
                'acousticness': 0.25,
                'loudness': -6.0,
                'tempo': 120.0
            }
            return defaults.get(feature_name, 0.5)
    
    def _get_actionable_advice(self, feature_name, current_value, target_value):
        """Generate actionable advice for improving a feature."""
        advice_templates = {
            'danceability': {
                'increase': f"Consider adding a more pronounced beat. Try increasing tempo slightly "
                           f"and emphasizing rhythmic elements. Target: {target_value:.2f}",
                'decrease': f"Reduce rhythmic emphasis with more varied tempo or less repetitive beats. "
                           f"Target: {target_value:.2f}"
            },
            'energy': {
                'increase': f"Boost intensity with louder dynamics, faster tempo, or more aggressive "
                           f"instrumentation. Target: {target_value:.2f}",
                'decrease': f"Soften intensity with quieter sections, acoustic instruments, or slower "
                           f"tempo. Target: {target_value:.2f}"
            },
            'valence': {
                'increase': f"Make the mood more positive with major chords, uplifting melodies, "
                           f"or brighter instrumentation. Target: {target_value:.2f}",
                'decrease': f"Add emotional depth with minor keys, slower tempo, or introspective "
                           f"themes. Target: {target_value:.2f}"
            },
            'acousticness': {
                'increase': f"Incorporate more acoustic instruments like guitar, piano, or unplugged "
                           f"arrangements. Target: {target_value:.2f}",
                'decrease': f"Add electronic elements like synths, drum machines, or digital effects. "
                           f"Target: {target_value:.2f}"
            },
            'loudness': {
                'increase': f"Increase overall volume and compression during mastering. "
                           f"Target: {target_value:.1f} dB",
                'decrease': f"Reduce volume and use more dynamic range. "
                           f"Target: {target_value:.1f} dB"
            },
            'tempo': {
                'increase': f"Speed up the tempo to create more energy. Target: {target_value:.0f} BPM",
                'decrease': f"Slow down for a more relaxed feel. Target: {target_value:.0f} BPM"
            }
        }
        
        direction = 'increase' if target_value > current_value else 'decrease'
        
        if feature_name in advice_templates:
            return advice_templates[feature_name][direction]
        else:
            return f"Adjust {feature_name} from {current_value:.2f} to {target_value:.2f}"
    
    def _assess_difficulty(self, feature_name, current_value, target_value):
        """Assess how difficult it is to make this change."""
        gap = abs(target_value - current_value)
        
        # Different features have different difficulty thresholds
        thresholds = {
            'danceability': (0.1, 0.2),
            'energy': (0.1, 0.2),
            'valence': (0.15, 0.3),
            'acousticness': (0.2, 0.4),
            'loudness': (3, 6),
            'tempo': (10, 20)
        }
        
        easy_threshold, medium_threshold = thresholds.get(feature_name, (0.1, 0.2))
        
        if gap < easy_threshold:
            return 'Easy'
        elif gap < medium_threshold:
            return 'Medium'
        else:
            return 'Hard'


# Test function
if __name__ == "__main__":
    from explainer import create_explainer
    import joblib
    
    # Load components
    model = joblib.load('../models/best_model_clean.pkl')
    explainer = create_explainer(
        '../models/best_model_clean.pkl',
        '../data/model_ready/X_train_clean.csv'
    )
    
    # Load reference data (use training data as proxy)
    reference_data = pd.read_csv('../data/model_ready/X_train_clean.csv')
    y_train = pd.read_csv('../data/model_ready/y_train.csv')
    reference_data['popularity'] = y_train['popularity'].values
    
    # Create recommender
    recommender = RecommenderModule(model, explainer, reference_data)
    
    # Test with sample
    test_data = pd.read_csv('../data/model_ready/X_test_clean.csv')
    sample = test_data.iloc[0:1]
    
    # Get current prediction
    current_pred = model.predict(sample)[0]
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(sample, current_pred)
    
    print("\n" + "="*60)
    print(f"Current Prediction: {current_pred:.1f}")
    print("\nRecommendations:")
    print("="*60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['feature'].upper()} ({rec['priority']} Priority)")
        print(f"   Current: {rec['current_value']:.2f} → Suggested: {rec['suggested_value']:.2f}")
        print(f"   Estimated Gain: +{rec['estimated_gain']:.1f} points")
        print(f"   Difficulty: {rec['difficulty']}")
        print(f"   Advice: {rec['actionable_advice']}")
```

---

## 🎨 STEP 5: Streamlit Page - Explanation (pages/4_💡_Explanation.py)

```python
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.explainer import create_explainer
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Explanation & Analysis", page_icon="💡", layout="wide")

# Load model and explainer (cached)
@st.cache_resource
def load_components():
    model = joblib.load("models/best_model_clean.pkl")
    explainer = create_explainer(
        "models/best_model_clean.pkl",
        "data/model_ready/X_train_clean.csv"
    )
    return model, explainer

model, explainer = load_components()

# Hero Section
st.markdown("""
<div style="background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
            border-radius: 28px; padding: 36px 40px; margin-bottom: 28px; color: white;">
    <h1 style="color: white; margin: 0;">💡 Tahmin Açıklaması & Analiz</h1>
    <p style="margin-top: 12px; opacity: 0.95;">
    Modeliniz <strong>NEDEN</strong> bu tahmini yaptı? SHAP analizi ile detaylı açıklama.
    </p>
</div>
""", unsafe_allow_html=True)

# Check if we have a prediction to explain
if 'last_prediction' not in st.session_state or st.session_state.last_prediction is None:
    st.warning("⚠️ Önce bir tahmin yapmalısınız!")
    st.info("👈 Sol menüden '🎯 Tekil Tahmin' sayfasına gidip bir tahmin yapın.")
    st.stop()

# Get last prediction data
input_features = st.session_state.get('last_input')
prediction = st.session_state.get('last_prediction')

if input_features is None:
    st.error("❌ Input features bulunamadı. Lütfen yeni bir tahmin yapın.")
    st.stop()

# Generate explanation
with st.spinner("🔍 Tahmin açıklanıyor..."):
    try:
        explanation_result = explainer.explain_prediction(input_features)
    except Exception as e:
        st.error(f"❌ Açıklama oluşturulurken hata: {str(e)}")
        st.stop()

# Display metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background: white; padding: 24px; border-radius: 20px; 
                box-shadow: 0 12px 35px rgba(31, 41, 55, 0.06);">
        <div style="font-size: 3rem; font-weight: 800; color: #1DB954;">
            {prediction:.1f}
        </div>
        <div style="color: #4B5563; font-size: 0.95rem; text-transform: uppercase; 
                    font-weight: 600; margin-top: 8px;">
            Predicted Popularity
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    confidence = explanation_result['confidence_score']
    confidence_color = "#06D6A0" if confidence > 70 else "#F18F01" if confidence > 50 else "#E63946"
    st.markdown(f"""
    <div style="background: white; padding: 24px; border-radius: 20px; 
                box-shadow: 0 12px 35px rgba(31, 41, 55, 0.06);">
        <div style="font-size: 3rem; font-weight: 800; color: {confidence_color};">
            {confidence:.1f}%
        </div>
        <div style="color: #4B5563; font-size: 0.95rem; text-transform: uppercase; 
                    font-weight: 600; margin-top: 8px;">
            Confidence Score
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    base_value = explanation_result['base_value']
    st.markdown(f"""
    <div style="background: white; padding: 24px; border-radius: 20px; 
                box-shadow: 0 12px 35px rgba(31, 41, 55, 0.06);">
        <div style="font-size: 3rem; font-weight: 800; color: #8E7DBE;">
            {base_value:.1f}
        </div>
        <div style="color: #4B5563; font-size: 0.95rem; text-transform: uppercase; 
                    font-weight: 600; margin-top: 8px;">
            Model Baseline
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Natural Language Explanation
st.markdown("### 📝 Açıklama")
st.markdown(f"""
<div style="background: white; border-left: 4px solid #1DB954; border-radius: 12px; 
            padding: 20px; margin: 16px 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);">
{explanation_result['explanation_text']}
</div>
""", unsafe_allow_html=True)

# Top Features Chart
st.markdown("### 🔝 En Etkili Özellikler")

top_features = explanation_result['top_features']

# Create horizontal bar chart
features = [f[0] for f in top_features]
impacts = [f[1] for f in top_features]
colors = ['#06D6A0' if imp > 0 else '#E63946' for imp in impacts]

fig = go.Figure(go.Bar(
    x=impacts,
    y=features,
    orientation='h',
    marker=dict(color=colors),
    text=[f"{imp:+.1f}" for imp in impacts],
    textposition='outside'
))

fig.update_layout(
    title="Feature Impact on Prediction (SHAP Values)",
    xaxis_title="Impact on Popularity",
    yaxis_title="",
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Detailed Feature Analysis
st.markdown("### 📊 Detaylı Özellik Analizi")

for feature_name, impact, value in top_features:
    with st.expander(f"{'✅' if impact > 0 else '⚠️'} {feature_name.title()} (Impact: {impact:+.1f})"):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Current Value", f"{value:.3f}")
            st.metric("Impact", f"{impact:+.1f} points", 
                     delta_color="normal" if impact > 0 else "inverse")
        
        with col2:
            if impact > 0:
                st.success(f"✅ This feature is **helping** your popularity prediction.")
                st.write(explainer._get_positive_reason(feature_name, value))
            else:
                st.warning(f"⚠️ This feature is **reducing** your popularity prediction.")
                st.write(explainer._get_negative_reason(feature_name, value))

# Next Steps
st.markdown("### 🚀 Sonraki Adımlar")

col1, col2 = st.columns(2)

with col1:
    if st.button("🎼 Öneri Motoru'na Git", use_container_width=True):
        st.switch_page("pages/5_🎼_Recommendations.py")

with col2:
    if st.button("🔬 Simülasyon Lab'a Git", use_container_width=True):
        st.switch_page("pages/6_🔬_Simulation_Lab.py")
```

---

## 📚 STEP 6: Update requirements.txt

```txt
# Add to existing requirements.txt
scipy>=1.11.0
shap>=0.44.0
plotly>=5.18.0
```

---

## 🎯 STEP 7: Quick Start Commands

```bash
# 1. Install new dependencies
pip install -r requirements.txt

# 2. Test explainer module
cd modules
python explainer.py

# 3. Test recommender module
python recommender.py

# 4. Run Streamlit app
cd ..
streamlit run app.py

# 5. Navigate to new "💡 Explanation" page
```

---

## 🔧 INTEGRATION CHECKLIST

- [ ] Create `modules/` directory
- [ ] Copy `explainer.py` to `modules/`
- [ ] Copy `recommender.py` to `modules/`
- [ ] Create `pages/4_💡_Explanation.py`
- [ ] Update `requirements.txt`
- [ ] Test explainer standalone
- [ ] Test recommender standalone
- [ ] Run Streamlit app
- [ ] Test explanation page
- [ ] Verify SHAP visualizations

---

**Ready to implement! Start with STEP 1 and work through sequentially.**

**Next:** Implement similarity engine and simulator modules.
