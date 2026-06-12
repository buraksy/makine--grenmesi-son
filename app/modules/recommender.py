"""
Recommender Module — şarkıyı Yüksek Popüler sınıfına yaklaştırma önerileri.
⚠️ Mevcut sürüm regresyon SHAP çıktılarına dayanır; sınıflandırma uyarlaması sırada.
Eşik: popularity >= 50 → Yüksek (1)
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
            reference_data: DataFrame of successful songs (with popularity column if available)
        """
        self.model = model
        self.explainer = explainer
        self.reference_data = reference_data
        
        # Filter successful songs (popularity >= 60) if popularity column exists
        if 'popularity' in reference_data.columns:
            self.successful_songs = reference_data[reference_data['popularity'] >= 60]
        else:
            # Use all reference data if no popularity column
            self.successful_songs = reference_data
        
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
        
        # Identify features with negative impact (now using dict format)
        negative_features = [
            (f['feature'], f['impact'], f['value']) 
            for f in top_features 
            if f['impact'] < -1.0  # Only significant negative impacts
        ]
        
        # If no negative features, find underperforming features
        if len(negative_features) == 0:
            negative_features = self._find_underperforming_features(
                input_features, genre
            )
        
        # Process more features to account for skipped ones
        for feature_name, current_impact, current_value in negative_features[:max_recommendations * 3]:
            # Skip genre features and boolean metadata (can't change those)
            if feature_name.startswith('genre_') or feature_name in ['explicit', 'key', 'mode', 'time_signature']:
                continue
            
            # Stop if we have enough recommendations
            if len(recommendations) >= max_recommendations:
                break
            
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
                'priority': 'Yüksek' if estimated_gain > 5 else 'Orta'
            })
        
        # Sort by potential impact
        recommendations.sort(key=lambda x: x['estimated_gain'], reverse=True)
        
        return recommendations[:max_recommendations]
    
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
                if feature_name in genre_songs.columns:
                    # Check if boolean column
                    if genre_songs[feature_name].dtype == bool or genre_songs[feature_name].nunique() == 2:
                        return genre_songs[feature_name].mode()[0] if len(genre_songs[feature_name].mode()) > 0 else 0
                    return genre_songs[feature_name].quantile(0.75)
        
        # Fallback to all successful songs
        if feature_name in self.successful_songs.columns:
            # Check if boolean column
            if self.successful_songs[feature_name].dtype == bool or self.successful_songs[feature_name].nunique() == 2:
                mode_values = self.successful_songs[feature_name].mode()
                return mode_values[0] if len(mode_values) > 0 else 0
            return self.successful_songs[feature_name].quantile(0.75)
        else:
            # Default values if feature not found
            defaults = {
                'danceability': 0.65,
                'energy': 0.60,
                'valence': 0.55,
                'acousticness': 0.25,
                'loudness': -6.0,
                'tempo': 120.0,
                'speechiness': 0.05,
                'instrumentalness': 0.00,
                'liveness': 0.15,
                'duration_ms': 200000
            }
            return defaults.get(feature_name, 0.5)
    
    def _get_actionable_advice(self, feature_name, current_value, target_value):
        """Generate actionable advice for improving a feature."""
        advice_templates = {
            'danceability': {
                'increase': f"Ritmi daha belirgin hale getirin. Tempo'yu hafifçe artırın ve ritmik öğeleri vurgulayın. Hedef: {target_value:.2f}",
                'decrease': f"Daha değişken tempo ve daha az tekrarlanan vuruşlarla ritmik vurguyu azaltın. Hedef: {target_value:.2f}"
            },
            'energy': {
                'increase': f"Daha yüksek dinamikler, daha hızlı tempo veya daha agresif enstrümantasyon ile yoğunluğu artırın. Hedef: {target_value:.2f}",
                'decrease': f"Sessiz bölümler, akustik enstrümanlar veya daha yavaş tempo ile yoğunluğu azaltın. Hedef: {target_value:.2f}"
            },
            'valence': {
                'increase': f"Majör akorlar, yükselten melodiler veya daha parlak enstrümantasyon ile ruh halini daha pozitif yapın. Hedef: {target_value:.2f}",
                'decrease': f"Minör tonlar, daha yavaş tempo veya içe dönük temalarla duygusal derinlik ekleyin. Hedef: {target_value:.2f}"
            },
            'acousticness': {
                'increase': f"Gitar, piyano veya unplugged aranjmanlar gibi daha fazla akustik enstrüman ekleyin. Hedef: {target_value:.2f}",
                'decrease': f"Synth'ler, davul makineleri veya dijital efektler gibi elektronik öğeler ekleyin. Hedef: {target_value:.2f}"
            },
            'loudness': {
                'increase': f"Mastering sırasında genel ses seviyesini ve sıkıştırmayı artırın. Hedef: {target_value:.1f} dB",
                'decrease': f"Ses seviyesini azaltın ve daha fazla dinamik aralık kullanın. Hedef: {target_value:.1f} dB"
            },
            'tempo': {
                'increase': f"Daha fazla enerji yaratmak için tempo'yu hızlandırın. Hedef: {target_value:.0f} BPM",
                'decrease': f"Daha rahat bir his için yavaşlatın. Hedef: {target_value:.0f} BPM"
            },
            'speechiness': {
                'increase': f"Daha fazla vokal içerik veya rap bölümleri ekleyin. Hedef: {target_value:.2f}",
                'decrease': f"Konuşma içeriğini azaltın, müzikal öğeleri artırın. Hedef: {target_value:.2f}"
            },
            'instrumentalness': {
                'increase': f"Vokal kullanımını azaltın, enstrümantal bölümleri artırın. Hedef: {target_value:.2f}",
                'decrease': f"Daha fazla vokal ekleyin, şarkı sözlerini öne çıkarın. Hedef: {target_value:.2f}"
            },
            'liveness': {
                'increase': f"Canlı performans hissi yaratın, kitle seslerini ekleyin. Hedef: {target_value:.2f}",
                'decrease': f"Stüdyo prodüksiyonu kalitesini artırın. Hedef: {target_value:.2f}"
            },
            'duration_ms': {
                'increase': f"Şarkıyı uzatın (ekstra bridge, solo vs.). Hedef: {target_value/1000:.0f} saniye",
                'decrease': f"Şarkıyı kısaltın, daha öz tutun. Hedef: {target_value/1000:.0f} saniye"
            }
        }
        
        direction = 'increase' if target_value > current_value else 'decrease'
        
        if feature_name in advice_templates:
            return advice_templates[feature_name][direction]
        else:
            return f"{feature_name} özelliğini {current_value:.2f}'den {target_value:.2f}'e ayarlayın"
    
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
            'tempo': (10, 20),
            'speechiness': (0.1, 0.2),
            'instrumentalness': (0.2, 0.4),
            'liveness': (0.1, 0.2),
            'duration_ms': (30000, 60000)
        }
        
        easy_threshold, medium_threshold = thresholds.get(feature_name, (0.1, 0.2))
        
        if gap < easy_threshold:
            return 'Kolay'
        elif gap < medium_threshold:
            return 'Orta'
        else:
            return 'Zor'
