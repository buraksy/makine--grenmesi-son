"""
PHASE 6: INSIGHT GENERATION
Amaç: Teknik sonuçları anlamlı içgörülere dönüştürmek
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# Klasörlerin varlığını garantile
Path('../reports/csv').mkdir(parents=True, exist_ok=True)
Path('../reports/markdown').mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 6: İÇGÖRÜ ÜRETİMİ (INSIGHT GENERATION)")
print("="*80)

# Tüm analiz sonuçlarını topla
try:
    phase1_summary = pd.read_csv('../reports/csv/phase1_data_overview_summary.csv')
    phase2_numeric = pd.read_csv('../reports/csv/phase2_univariate_numeric_summary.csv')
    phase2_categorical = pd.read_csv('../reports/csv/phase2_univariate_categorical_summary.csv')
    phase4_correlation = pd.read_csv('../reports/csv/phase4_correlation_matrix.csv')
    phase4_vif = pd.read_csv('../reports/csv/phase4_vif_analysis.csv')
    phase5_outliers = pd.read_csv('../reports/csv/phase5_outlier_analysis.csv')
    
    print("✓ Tüm analiz raporları başarıyla yüklendi")
except Exception as e:
    print(f"⚠ Bazı analiz raporları yüklenemedi: {e}")

print("\n" + "="*80)
print("TEMEL İÇGÖRÜLER")
print("="*80)

insights = []

# İçgörü 1: Veri Seti Büyüklüğü
insight_1 = {
    "Sıra": 1,
    "İçgörü Başlığı": "Büyük Ölçekli Müzik Veri Seti",
    "Kanıt": "114,000 şarkı, 21 değişken, 9 audio feature",
    "İş Değeri": "Geniş veri seti, makine öğrenmesi modelleri için yeterli örnek sayısı sağlıyor",
    "Modelleme Etkisi": "Yüksek - Büyük veri seti overfitting riskini azaltır ve genelleme gücünü artırır"
}
insights.append(insight_1)

# İçgörü 2: Popularity Dağılımının Dengeli Yapısı
insight_2 = {
    "Sıra": 2,
    "İçgörü Başlığı": "Dengeli Popularity Dağılımı - Regression Problemi",
    "Kanıt": "Popularity ortalaması 33.2, medyan 35.0, skewness 0.046 (neredeyse simetrik)",
    "İş Değeri": "Hedef değişken dengeli ve simetrik dağılmış - hem popüler hem niche şarkılar dengeli temsil ediliyor",
    "Modelleme Etkisi": "Orta - Dengeli dağılım regression modellemesi için uygun, ancak extreme değerler az (%0.84 çok yüksek popularity)"
}
insights.append(insight_2)

# İçgörü 3: KRİTİK BULGU - Audio Features ile Popularity Arasında Zayıf İlişki
insight_3 = {
    "Sıra": 3,
    "İçgörü Başlığı": "⚠️ KRİTİK: Audio Features, Popularity'yi Açıklamıyor",
    "Kanıt": "En yüksek korelasyon loudness ile 0.0504. Tüm audio features ile popularity arası korelasyon <0.1",
    "İş Değeri": "Şarkının popülerliği, müzikal özelliklerden ziyade external faktörlerle (sanatçı ünü, pazarlama, playlist yerleştirme) belirleniyor",
    "Modelleme Etkisi": "ÇOK YÜKSEK RİSK - Mevcut feature'lar ile yüksek accuracy beklenmemeli. R² < 0.10 olası. Alternatif feature'lar (sanatçı, tür, yayın tarihi) gerekli"
}
insights.append(insight_3)

# İçgörü 3: Yüksek Multicollinearity Riski
if 'phase4_vif' in locals():
    high_vif_features = phase4_vif[phase4_vif['VIF'] > 10]
    if len(high_vif_features) > 0:
        insight_3 = {
            "Sıra": 3,
            "İçgörü Başlığı": "Multicollinearity Riski - Feature Engineering Fırsatı",
            "Kanıt": f"{len(high_vif_features)} değişkende VIF > 10 (tempo: 15.2, energy: 15.0, danceability: 12.3)",
            "İş Değeri": "Bazı müzik özellikleri birbirleriyle yüksek korelasyon gösteriyor - özellik seçimi gerekli",
            "Modelleme Etkisi": "Orta - Regularization (Ridge/Lasso) veya PCA ile multicollinearity yönetilebilir"
        }
        insights.append(insight_3)

# İçgörü 4: Outlier Yoğunluğu
if 'phase5_outliers' in locals():
    high_outliers = phase5_outliers[phase5_outliers['Outlier Oranı (%)'] > 10]
    if len(high_outliers) > 0:
        insight_4 = {
            "Sıra": 4,
            "İçgörü Başlığı": "Yüksek Outlier Oranı - Veri Hazırlama Gereksinimi",
            "Kanıt": f"instrumentalness (%22.15), speechiness (%11.59), time_signature (%10.66) değişkenlerinde yüksek outlier",
            "İş Değeri": "Bazı müzik özellikleri uç değerler içeriyor - bu aykırılıklar türe özgü özellikler olabilir",
            "Modelleme Etkisi": "Orta - Robust scaler veya outlier treatment, model performansını iyileştirebilir"
        }
        insights.append(insight_4)

# İçgörü 5: Minimal Veri Kalitesi Sorunu
insight_5 = {
    "Sıra": 5,
    "İçgörü Başlığı": "Yüksek Veri Kalitesi - Minimal Temizlik Gereksinimi",
    "Kanıt": "Eksik veri %0.0, duplicate satır yok, mantıksal tutarsızlık yok",
    "İş Değeri": "Veri seti temiz ve kullanıma hazır - veri hazırlama maliyeti düşük",
    "Modelleme Etkisi": "Yüksek - Hemen modelleme aşamasına geçilebilir, sadece feature engineering odaklı çalışma gerekli"
}
insights.append(insight_5)

# İçgörü 6: Genre ile Popularity İlişkisi
insight_6 = {
    "Sıra": 6,
    "İçgörü Başlığı": "Tür Bazlı Popularity Farklılaşması",
    "Kanıt": "Pop-film (59.3), k-pop (56.9), chill (53.7) en yüksek ortalama popularity. Acoustic (20-25) düşük.",
    "İş Değeri": "Müzik türü, popularity'nin en güçlü açıklayıcı değişkeni olabilir (audio features'tan daha etkili)",
    "Modelleme Etkisi": "Yüksek - track_genre kategorik encoding ile feature olarak kullanılmalı. One-hot veya target encoding ile model performansı artırılabilir"
}
insights.append(insight_6)

# İçgörü 7: Energy-Loudness-Acousticness İlişkisi
insight_7 = {
    "Sıra": 7,
    "İçgörü Başlığı": "Müzik Özellikleri Arasında Güçlü İlişkiler",
    "Kanıt": "energy-loudness korelasyonu 0.76, energy-acousticness korelasyonu -0.73",
    "İş Değeri": "Enerjili şarkılar daha gürültülü ve daha az akustik - tutarlı müzik karakterizasyonu",
    "Modelleme Etkisi": "Orta - Feature engineering ile bu ilişkiler composite feature olarak kullanılabilir"
}
insights.append(insight_7)

# İçgörü 8: Explicit İçerik Etkisi
insight_8 = {
    "Sıra": 8,
    "İçgörü Başlığı": "Explicit İçerik ile Popularity Arasında Pozitif İlişki",
    "Kanıt": "Explicit şarkıların ortalama popularity'si 36.5, explicit olmayan 32.9 (%11 fark)",
    "İş Değeri": "Explicit içerik, popularity üzerinde küçük ama gözlemlenebilir pozitif etkiye sahip",
    "Modelleme Etkisi": "Düşük-Orta - explicit feature olarak modele dahil edilmeli, ancak tek başına güçlü açıklayıcı değil"
}
insights.append(insight_8)

# İçgörüleri kaydet
insights_df = pd.DataFrame(insights)
insights_df.to_csv('../reports/csv/phase6_key_insights.csv', index=False, encoding='utf-8-sig')
print("\n✓ Temel içgörüler kaydedildi: ../reports/csv/phase6_key_insights.csv")

print("\n" + "="*80)
print("KRİTİK DEĞİŞKENLER")
print("="*80)

# Modelleme için kritik değişkenler (REGRESSION problemi - popularity tahmini)
critical_features = {
    "Değişken": [
        "popularity",
        "track_genre",
        "explicit",
        "loudness",
        "danceability",
        "speechiness",
        "energy",
        "valence",
        "acousticness",
        "tempo"
    ],
    "Önem Derecesi": [
        "Hedef",
        "ÇOK YÜKSEK",
        "Orta",
        "Düşük",
        "Düşük",
        "Düşük",
        "Düşük",
        "Düşük",
        "Düşük",
        "Düşük"
    ],
    "Kullanım Amacı": [
        "Tahmin edilecek hedef değişken (0-100 arası popularity skoru)",
        "EN GÜÇLÜ FEATURE - Tür bilgisi popularity'yi en iyi açıklıyor (kategorik encoding gerekli)",
        "Explicit içerik, popularity ile pozitif ilişkili (%11 fark)",
        "En yüksek korelasyon (0.05) - çok zayıf ama en iyi audio feature",
        "Zayıf korelasyon (0.04) - dans edilebilirlik popularity ile ilişkisiz",
        "Zayıf negatif korelasyon (-0.04) - konuşma içeriği popularity düşürüyor",
        "Korelasyon ~0 - enerji popularity'yi açıklamıyor",
        "Korelasyon ~0 - pozitif ton popularity'yi açıklamıyor",
        "Korelasyon ~0 - akustiklik popularity'yi açıklamıyor",
        "Korelasyon ~0 - tempo popularity'yi açıklamıyor"
    ]
}

critical_df = pd.DataFrame(critical_features)
critical_df.to_csv('../reports/csv/phase6_critical_features.csv', index=False, encoding='utf-8-sig')
print("\n✓ Kritik değişkenler kaydedildi: ../reports/csv/phase6_critical_features.csv")

print(critical_df)

print("\n" + "="*80)
print("FEATURE ENGINEERING FIRSATLARı")
print("="*80)

feature_engineering_opportunities = [
    {
        "Fırsat": "Genre One-Hot Encoding veya Target Encoding",
        "Açıklama": "track_genre kategorik değişkeni popularity'nin EN GÜÇLÜ açıklayıcısı - mutlaka encode edilmeli",
        "Öncelik": "ÇOK YÜKSEK (KRİTİK)"
    },
    {
        "Fırsat": "Artist/Album Features (Harici Veri)",
        "Açıklama": "Sanatçı takipçi sayısı, önceki şarkı başarıları, playlist sayısı gibi external features eklenebilir",
        "Öncelik": "ÇOK YÜKSEK (audio features yetersiz)"
    },
    {
        "Fırsat": "Temporal Features (Yayın Tarihi)",
        "Açıklama": "Şarkı yayın tarihi, trend dönemleri, sezonsal etkiler popularity'yi etkileyebilir",
        "Öncelik": "Yüksek"
    },
    {
        "Fırsat": "Audio Feature Polynomials",
        "Açıklama": "loudness², danceability², feature interactions (her ne kadar zayıf olsalar da non-linear ilişki test edilebilir)",
        "Öncelik": "Düşük (lineer ilişki zaten yok)"
    },
    {
        "Fırsat": "Mood Score (valence x energy)",
        "Açıklama": "Şarkının genel mood'unu temsil eden composite feature (her ne kadar popularity ile ilişki zayıf olsa da)",
        "Öncelik": "Düşük"
    }
]

fe_df = pd.DataFrame(feature_engineering_opportunities)
fe_df.to_csv('../reports/csv/phase6_feature_engineering_opportunities.csv', index=False, encoding='utf-8-sig')
print("\n✓ Feature engineering fırsatları kaydedildi:")
print("  ../reports/csv/phase6_feature_engineering_opportunities.csv")

print(fe_df)

print("\n" + "="*80)
print("TÜM DATA PREP ÖNERİLERİNİ TOPLA")
print("="*80)

# Tüm phase'lerden Data Prep önerilerini topla
all_recommendations = []

for phase_num in [2, 4, 5]:
    file_path = f'../reports/csv/phase{phase_num}_data_prep_recommendations.csv'
    if os.path.exists(file_path):
        phase_recs = pd.read_csv(file_path, encoding='utf-8-sig')
        phase_recs['Phase'] = f'Phase {phase_num}'
        all_recommendations.append(phase_recs)

if len(all_recommendations) > 0:
    consolidated_recs = pd.concat(all_recommendations, ignore_index=True)
    
    # Öncelik sırasına göre sırala
    priority_order = {'Yüksek': 1, 'Orta': 2, 'Düşük': 3}
    consolidated_recs['Priority_Order'] = consolidated_recs['Öncelik'].map(priority_order)
    consolidated_recs = consolidated_recs.sort_values('Priority_Order').drop('Priority_Order', axis=1)
    
    consolidated_recs.to_csv('../reports/csv/phase6_all_data_prep_recommendations.csv', index=False, encoding='utf-8-sig')
    print("\n✓ Tüm Data Prep önerileri konsolide edildi:")
    print("  ../reports/csv/phase6_all_data_prep_recommendations.csv")
    print(f"\nToplam öneri sayısı: {len(consolidated_recs)}")
    print(f"Yüksek öncelikli: {len(consolidated_recs[consolidated_recs['Öncelik'] == 'Yüksek'])}")
    print(f"Orta öncelikli: {len(consolidated_recs[consolidated_recs['Öncelik'] == 'Orta'])}")
    print(f"Düşük öncelikli: {len(consolidated_recs[consolidated_recs['Öncelik'] == 'Düşük'])}")

print("\n" + "="*80)
print("PHASE 6 TAMAMLANDI")
print("="*80)
