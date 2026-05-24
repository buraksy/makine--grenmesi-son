"""
Explainer Module - WHY did the model predict this?
SHAP-based explainability for music popularity predictions
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
        
        # Handle both DataFrame and numpy array
        if isinstance(background_data, pd.DataFrame):
            background_sample = background_data.sample(min(100, len(background_data)), random_state=42).values
        else:
            # If numpy array, use numpy random sampling
            n_samples = min(100, len(background_data))
            np.random.seed(42)
            indices = np.random.choice(len(background_data), n_samples, replace=False)
            background_sample = background_data[indices]
        
        self.explainer = shap.KernelExplainer(
            model.predict, 
            background_sample
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
        # Get absolute impacts with dictionary format
        impacts = [
            {
                'feature': self.feature_names[i],
                'impact': float(shap_values[i]),
                'value': float(input_values[i])
            }
            for i in range(len(shap_values))
        ]
        
        # Sort by absolute impact
        impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return impacts[:top_k]
    
    def _generate_explanation(self, prediction, top_features, input_features):
        """Generate natural language explanation."""
        explanation = f"🎵 **Predicted Popularity: {prediction:.1f}/100**\n\n"
        explanation += "**Main Drivers:**\n\n"
        
        for feature_dict in top_features:
            feature_name = feature_dict['feature']
            impact = feature_dict['impact']
            value = feature_dict['value']
            
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
            'danceability': "Yüksek dansedilebilirlik ana akım dinleyicileri çeker",
            'energy': "Güçlü enerji etkileşim yaratır",
            'valence': "Pozitif ruh hali geniş kitlelere hitap eder",
            'acousticness': "Akustik öğeler otantiklik katar",
            'loudness': "Streaming platformları için iyi ses seviyesi",
            'tempo': "Bu tempo tür için uygun",
            'speechiness': "Bu speechiness seviyesi dengeli",
            'instrumentalness': "Enstrümantal öğeler dengeli",
            'liveness': "Canlılık hissi olumlu etki yaratıyor"
        }
        
        # Handle genre features
        if feature_name.startswith('track_genre_'):
            return "Bu tür popülerlik potansiyeline sahip"
        
        return reasons.get(feature_name, f"Bu {feature_name} değeri olumlu katkı sağlıyor")
    
    def _get_negative_reason(self, feature_name, value):
        """Get reason for negative impact."""
        reasons = {
            'speechiness': "Yüksek konuşma içeriği müzikal çekiciliği sınırlayabilir",
            'instrumentalness': "Vokal eksikliği ana akım çekiciliği azaltabilir",
            'liveness': "Canlı kayıt kalitesi prodüksiyon değerini etkileyebilir",
            'acousticness': "Düşük akustik varlık sıcaklık eksikliği yaratabilir",
            'duration_ms': "Şarkı uzunluğu streaming için optimal olmayabilir",
            'danceability': "Düşük dansedilebilirlik etkileşimi azaltabilir",
            'energy': "Düşük enerji dinleyici ilgisini azaltabilir",
            'valence': "Negatif ruh hali kitlenizi sınırlayabilir",
            'loudness': "Ses seviyesi optimal değil",
            'tempo': "Tempo optimal değil"
        }
        
        # Handle genre features
        if feature_name.startswith('track_genre_'):
            return "Bu tür daha düşük popülerlik eğiliminde"
        
        return reasons.get(feature_name, f"Bu {feature_name} değeri olumsuz etki yaratıyor")
    
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
