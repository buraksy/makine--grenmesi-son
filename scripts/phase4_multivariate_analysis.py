"""
PHASE 4: MULTIVARIATE ANALYSIS
Amaç: Çok değişkenli yapıyı ve birlikte hareket eden değişkenleri incelemek
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Klasörlerin varlığını garantile
Path('../figures').mkdir(parents=True, exist_ok=True)
Path('../reports/csv').mkdir(parents=True, exist_ok=True)

# Profesyonel renk paleti
PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
    "#BC4B51", "#8E7DBE", "#F77F00", "#06A77D", "#D4A574"
]

# Data Prep önerileri
data_prep_recommendations = []

def add_data_prep_recommendation(issue, evidence, recommendation, priority="Orta"):
    data_prep_recommendations.append({
        "Sorun": issue,
        "Kanıt": evidence,
        "Öneri": recommendation,
        "Öncelik": priority
    })

def apply_premium_layout(fig, title):
    """Profesyonel, net ve görkemli grafik düzeni uygular"""
    fig.update_layout(
        title={
            "text": title,
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 24, "family": "Arial Black", "color": "#1F2937", "weight": "bold"}
        },
        template="plotly_white",
        paper_bgcolor="#FBFBF8",
        plot_bgcolor="#FBFBF8",
        font={"family": "Arial", "size": 13, "color": "#374151"},
        margin=dict(l=60, r=40, t=80, b=60),
        legend_title_text="Kategori",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig

def save_figure(fig, file_base):
    html_path = f"../figures/{file_base}.html"
    fig.write_html(html_path)
    print(f"  ✓ Grafik kaydedildi: {html_path}")
    
    try:
        png_path = f"../figures/{file_base}.png"
        fig.write_image(png_path)
        print(f"  ✓ Grafik kaydedildi: {png_path}")
    except:
        print(f"  ⚠ PNG kaydı yapılamadı (kaleido gerekebilir)")

print("="*80)
print("PHASE 4: ÇOK DEĞİŞKENLİ ANALİZ (MULTIVARIATE ANALYSIS)")
print("="*80)

# Veri setini yükle
df = pd.read_csv('../data/raw/dataset.csv')

# Sayısal değişkenler
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Unnamed: 0' in numeric_cols:
    numeric_cols.remove('Unnamed: 0')

# Müzik özellikleri (audio features)
audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

print("\n" + "="*80)
print("A. KORELASYON MATRİSİ ANALİZİ")
print("="*80)

# Tüm sayısal değişkenler için korelasyon
correlation_matrix = df[numeric_cols].corr()

print("\nKorelasyon Matrisi Hesaplandı")
print(f"Boyut: {correlation_matrix.shape}")

# Yüksek korelasyonları tespit et
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
            high_corr_pairs.append({
                'Değişken 1': correlation_matrix.columns[i],
                'Değişken 2': correlation_matrix.columns[j],
                'Korelasyon': round(corr_value, 4)
            })

if len(high_corr_pairs) > 0:
    print(f"\n⚠ Yüksek Korelasyon Tespit Edildi (|r| > 0.7):")
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(high_corr_df)
    
    for pair in high_corr_pairs:
        add_data_prep_recommendation(
            issue=f"Yüksek korelasyon: {pair['Değişken 1']} - {pair['Değişken 2']}",
            evidence=f"Korelasyon katsayısı {pair['Korelasyon']:.4f} olarak hesaplandı.",
            recommendation="Data Prep Expert; değişken eleme, PCA veya feature selection teknikleri değerlendirmelidir.",
            priority="Yüksek" if abs(pair['Korelasyon']) > 0.8 else "Orta"
        )
else:
    print("\n✓ Kritik seviyede yüksek korelasyon tespit edilmedi.")

# Korelasyon heatmap
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu_r',
    zmid=0,
    text=correlation_matrix.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 8},
    colorbar=dict(title="Korelasyon")
))

fig = apply_premium_layout(fig, "Sayısal Değişkenler Korelasyon Matrisi")
fig.update_xaxes(tickangle=45)
print(f"\nGörselleştirme: Korelasyon Heatmap")
save_figure(fig, "phase4_correlation_matrix_full")

# Audio features için ayrı heatmap
audio_corr = df[audio_features].corr()

fig = go.Figure(data=go.Heatmap(
    z=audio_corr.values,
    x=audio_corr.columns,
    y=audio_corr.columns,
    colorscale='RdBu_r',
    zmid=0,
    text=audio_corr.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar=dict(title="Korelasyon")
))

fig = apply_premium_layout(fig, "Müzik Özellikleri Korelasyon Matrisi")
fig.update_xaxes(tickangle=45)
print(f"\nGörselleştirme: Audio Features Korelasyon Heatmap")
save_figure(fig, "phase4_correlation_matrix_audio_features")

# Korelasyon matrisini CSV'ye kaydet
correlation_matrix.to_csv('../reports/csv/phase4_correlation_matrix.csv')
print(f"\n✓ Korelasyon matrisi kaydedildi: ../reports/csv/phase4_correlation_matrix.csv")

print("\n" + "="*80)
print("B. VIF (VARIANCE INFLATION FACTOR) ANALİZİ")
print("="*80)

# VIF hesaplama - sadece audio features için (performans için)
print("\nMüzik Özellikleri için VIF Hesaplanıyor...")

vif_data = []
df_vif = df[audio_features].dropna()

for i, col in enumerate(audio_features):
    try:
        vif_value = variance_inflation_factor(df_vif.values, i)
        vif_data.append({
            'Değişken': col,
            'VIF': round(vif_value, 4)
        })
        print(f"  {col}: VIF = {vif_value:.4f}")
    except Exception as e:
        print(f"  {col}: VIF hesaplanamadı ({str(e)})")
        vif_data.append({
            'Değişken': col,
            'VIF': None
        })

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

print(f"\nVIF Özet:")
print(vif_df)

# VIF görselleştirme
vif_plot_df = vif_df[vif_df['VIF'].notna()].copy()

if len(vif_plot_df) > 0:
    fig = px.bar(
        vif_plot_df,
        y='Değişken',
        x='VIF',
        orientation='h',
        color='VIF',
        color_continuous_scale=[(0, PROFESSIONAL_PALETTE[4]), (0.5, PROFESSIONAL_PALETTE[1]), (1, PROFESSIONAL_PALETTE[3])],
        title="Variance Inflation Factor (VIF) - Müzik Özellikleri"
    )
    
    # VIF = 5 ve 10 referans çizgileri
    fig.add_vline(x=5, line_dash="dash", line_color="orange", annotation_text="VIF=5 (Orta Risk)")
    fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="VIF=10 (Yüksek Risk)")
    
    fig = apply_premium_layout(fig, "Variance Inflation Factor (VIF) - Müzik Özellikleri")
    print(f"\nGörselleştirme: VIF Bar Chart")
    save_figure(fig, "phase4_vif_analysis")

# VIF > 10 olanlar için öneri
high_vif = vif_df[vif_df['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠ Yüksek VIF Tespit Edildi (VIF > 10):")
    print(high_vif)
    
    for _, row in high_vif.iterrows():
        add_data_prep_recommendation(
            issue=f"Yüksek VIF: {row['Değişken']}",
            evidence=f"VIF değeri {row['VIF']:.4f} olarak hesaplandı (>10).",
            recommendation="Modeling Expert; regularization (Ridge, Lasso) veya değişken eleme seçeneklerini değerlendirmelidir.",
            priority="Yüksek"
        )
else:
    print("\n✓ Kritik seviyede multicollinearity tespit edilmedi (VIF < 10).")

# VIF tablosunu kaydet
vif_df.to_csv('../reports/csv/phase4_vif_analysis.csv', index=False)
print(f"\n✓ VIF analizi kaydedildi: ../reports/csv/phase4_vif_analysis.csv")

print("\n" + "="*80)
print("C. DEĞİŞKEN İLİŞKİ GRAFİĞİ (PAIRWISE RELATIONSHIPS)")
print("="*80)

# Önemli 4 müzik özelliği için scatter matrix
key_features = ['danceability', 'energy', 'valence', 'acousticness']

print(f"\nAnahtar Müzik Özellikleri için Scatter Matrix:")
print(f"  {', '.join(key_features)}")

# Sample al (performans için)
sample_size = min(2000, len(df))
df_sample = df[key_features].sample(n=sample_size, random_state=42)

fig = px.scatter_matrix(
    df_sample,
    dimensions=key_features,
    color_discrete_sequence=[PROFESSIONAL_PALETTE[0]],
    title="Anahtar Müzik Özellikleri - Pairwise Scatter Matrix",
    opacity=0.4
)

fig.update_traces(diagonal_visible=False, showupperhalf=False)
fig = apply_premium_layout(fig, "Anahtar Müzik Özellikleri - Pairwise Scatter Matrix")
print(f"\nGörselleştirme: Scatter Matrix")
save_figure(fig, "phase4_scatter_matrix_key_features")

# Data Prep önerilerini kaydet
if len(data_prep_recommendations) > 0:
    recommendations_df = pd.DataFrame(data_prep_recommendations)
    recommendations_df.to_csv('../reports/csv/phase4_data_prep_recommendations.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ Data Prep Expert önerileri kaydedildi:")
    print(f"  ../reports/csv/phase4_data_prep_recommendations.csv")

print("\n" + "="*80)
print("PHASE 4 TAMAMLANDI")
print("="*80)
