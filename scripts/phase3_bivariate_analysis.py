"""
PHASE 3: BIVARIATE ANALYSIS
Amaç: Değişkenler arasındaki ikili ilişkileri incelemek
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

# Klasörlerin varlığını garantile
Path('../figures').mkdir(parents=True, exist_ok=True)
Path('../reports/csv').mkdir(parents=True, exist_ok=True)

# Profesyonel renk paleti
PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
    "#BC4B51", "#8E7DBE", "#F77F00", "#06A77D", "#D4A574"
]

print("="*80)
print("PHASE 3: İKİLİ DEĞİŞKEN ANALİZİ (BIVARIATE ANALYSIS)")
print("="*80)

# Veri setini yükle
df = pd.read_csv('../data/raw/dataset.csv')

# Hedef değişken olarak popularity kullanacağız (Regression problemi)
target_col = 'popularity'
print(f"\nHedef Değişken: {target_col}")
print(f"Değer Aralığı: {df[target_col].min():.0f} - {df[target_col].max():.0f}")
print(f"Ortalama: {df[target_col].mean():.2f}")
print(f"Medyan: {df[target_col].median():.2f}")

# Değişken tipleri
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Unnamed: 0' in numeric_cols:
    numeric_cols.remove('Unnamed: 0')

categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

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

print("\n" + "="*80)
print("A. HEDEF DEĞİŞKEN DAĞILIMI (POPULARITY)")
print("="*80)

# Hedef değişken dağılımı (popularity - sürekli değişken)
print(f"\nPopularity İstatistikleri:")
print(f"  Min: {df[target_col].min():.0f}")
print(f"  Max: {df[target_col].max():.0f}")
print(f"  Ortalama: {df[target_col].mean():.2f}")
print(f"  Medyan: {df[target_col].median():.2f}")
print(f"  Std Sapma: {df[target_col].std():.2f}")
print(f"  Skewness: {df[target_col].skew():.4f}")

# Histogram
fig = px.histogram(
    df,
    x=target_col,
    nbins=50,
    color_discrete_sequence=[PROFESSIONAL_PALETTE[0]],
    title=f"{target_col} - Dağılım Histogramı"
)
fig = apply_premium_layout(fig, f"{target_col} - Dağılım Histogramı")
fig.update_traces(marker_line_width=1, marker_line_color="#FFFFFF")
print(f"\nGörselleştirme: Popularity Histogram")
save_figure(fig, f"phase3_target_distribution_{target_col}")

# Popularity grupları analizi
bins = [0, 20, 40, 60, 80, 100]
labels = ['Çok Düşük (0-20)', 'Düşük (20-40)', 'Orta (40-60)', 'Yüksek (60-80)', 'Çok Yüksek (80-100)']
df['popularity_group'] = pd.cut(df[target_col], bins=bins, labels=labels, include_lowest=True)

pop_group_counts = df['popularity_group'].value_counts().sort_index()
pop_group_ratio = (pop_group_counts / len(df) * 100).round(2)

print(f"\nPopularity Grupları Dağılımı:")
for label, count, ratio in zip(pop_group_counts.index, pop_group_counts.values, pop_group_ratio.values):
    print(f"  {label}: {count} şarkı (%{ratio:.2f})")

# Bar chart - Popularity grupları
plot_data = pd.DataFrame({
    'Grup': pop_group_counts.index,
    'Şarkı Sayısı': pop_group_counts.values,
    'Oran (%)': pop_group_ratio.values
})

fig = px.bar(
    plot_data,
    x='Grup',
    y='Oran (%)',
    color='Oran (%)',
    color_continuous_scale=[(0, PROFESSIONAL_PALETTE[0]), (1, PROFESSIONAL_PALETTE[1])],
    title="Popularity Gruplarına Göre Dağılım"
)
fig = apply_premium_layout(fig, "Popularity Gruplarına Göre Dağılım")
fig.update_xaxes(tickangle=45)
print(f"\nGörselleştirme: Popularity Grupları")
save_figure(fig, f"phase3_popularity_groups")

print("\n" + "="*80)
print("B. SAYISAL DEĞİŞKENLER vs POPULARITY (KORELASYON ANALİZİ)")
print("="*80)

bivariate_numeric_results = []

for col in numeric_cols[:8]:  # İlk 8 sayısal değişken (zamandan tasarruf)
    if col == target_col:  # popularity'nin kendisini atla
        continue
        
    print(f"\n{'='*80}")
    print(f"Değişken: {col} vs {target_col}")
    print(f"{'='*80}")
    
    # Korelasyon hesapla
    correlation = df[[col, target_col]].corr().iloc[0, 1]
    
    print(f"\nKorelasyon: {correlation:.4f}")
    
    if abs(correlation) > 0.3:
        strength = "Güçlü" if abs(correlation) > 0.5 else "Orta"
        direction = "Pozitif" if correlation > 0 else "Negatif"
        print(f"  → {strength} {direction} ilişki")
    else:
        print(f"  → Zayıf ilişki")
    
    # Scatter plot
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    fig = px.scatter(
        df_sample,
        x=col,
        y=target_col,
        trendline="ols",
        color_discrete_sequence=[PROFESSIONAL_PALETTE[0]],
        title=f"{col} vs {target_col} (Korelasyon: {correlation:.3f})",
        opacity=0.5
    )
    fig = apply_premium_layout(fig, f"{col} vs {target_col} (Korelasyon: {correlation:.3f})")
    print(f"\nGörselleştirme: Scatter Plot")
    save_figure(fig, f"phase3_scatter_{col}_vs_{target_col}")
    
    bivariate_numeric_results.append({
        'Değişken': col,
        'Korelasyon': round(correlation, 4),
        'İlişki Gücü': 'Güçlü' if abs(correlation) > 0.5 else ('Orta' if abs(correlation) > 0.3 else 'Zayıf'),
        'İlişki Yönü': 'Pozitif' if correlation > 0 else 'Negatif'
    })

# Özet tablosunu kaydet
bivariate_numeric_df = pd.DataFrame(bivariate_numeric_results)
bivariate_numeric_df.to_csv('../reports/csv/phase3_bivariate_numeric_summary.csv', index=False)
print(f"\n{'='*80}")
print("✓ Sayısal değişkenler bivariate özet tablosu kaydedildi:")
print("  ../reports/csv/phase3_bivariate_numeric_summary.csv")

print("\n" + "="*80)
print("C. KATEGORİK DEĞİŞKENLER vs POPULARITY")
print("="*80)

# explicit değişkenini inceleyelim
if 'explicit' in categorical_cols:
    print(f"\n{'='*80}")
    print(f"Değişken: explicit vs {target_col}")
    print(f"{'='*80}")
    
    # Explicit gruplarına göre popularity ortalaması
    explicit_pop = df.groupby('explicit')[target_col].agg(['mean', 'median', 'std', 'count'])
    
    print(f"\nExplicit Gruplara Göre Popularity:")
    print(explicit_pop)
    
    # Boxplot
    fig = px.box(
        df,
        x='explicit',
        y=target_col,
        color='explicit',
        color_discrete_sequence=[PROFESSIONAL_PALETTE[0], PROFESSIONAL_PALETTE[3]],
        title=f"Explicit İçeriğe Göre {target_col} Dağılımı"
    )
    fig.update_layout(showlegend=False)
    fig = apply_premium_layout(fig, f"Explicit İçeriğe Göre {target_col} Dağılımı")
    print(f"\nGörselleştirme: Explicit Boxplot")
    save_figure(fig, f"phase3_boxplot_explicit_vs_{target_col}")

# track_genre değişkenini inceleyelim (en popüler 10 tür)
if 'track_genre' in categorical_cols:
    print(f"\n{'='*80}")
    print(f"Değişken: track_genre vs {target_col}")
    print(f"{'='*80}")
    
    # En popüler 10 türü seç
    top_genres = df.groupby('track_genre')[target_col].mean().sort_values(ascending=False).head(10)
    
    print(f"\nEn Yüksek Ortalama Popularity'ye Sahip 10 Tür:")
    print(top_genres)
    
    # Bar chart
    plot_data = top_genres.reset_index()
    plot_data.columns = ['Tür', 'Ortalama Popularity']
    
    fig = px.bar(
        plot_data,
        y='Tür',
        x='Ortalama Popularity',
        orientation='h',
        color='Ortalama Popularity',
        color_continuous_scale=[(0, PROFESSIONAL_PALETTE[0]), (1, PROFESSIONAL_PALETTE[1])],
        title=f"En Yüksek Ortalama Popularity - İlk 10 Tür"
    )
    fig = apply_premium_layout(fig, f"En Yüksek Ortalama Popularity - İlk 10 Tür")
    print(f"\nGörselleştirme: Türlere Göre Popularity")
    save_figure(fig, f"phase3_bar_genre_vs_{target_col}")

print("\n" + "="*80)
print("D. EN GÜÇLÜ İLİŞKİLER ÖZETİ")
print("="*80)

# Tüm korelasyonları sırala
if len(bivariate_numeric_results) > 0:
    bivariate_df = pd.DataFrame(bivariate_numeric_results)
    bivariate_df['Abs_Korelasyon'] = bivariate_df['Korelasyon'].abs()
    bivariate_df_sorted = bivariate_df.sort_values('Abs_Korelasyon', ascending=False)
    
    print(f"\nPopularity ile En Güçlü İlişkiye Sahip Değişkenler:")
    print(bivariate_df_sorted[['Değişken', 'Korelasyon', 'İlişki Gücü']].head(10))
    
    # En güçlü 3 ilişki için scatter plot
    top_3_features = bivariate_df_sorted.head(3)['Değişken'].tolist()
    
    for feature in top_3_features:
        if feature in df.columns:
            corr_val = df[[feature, target_col]].corr().iloc[0, 1]
            
            # Popularity gruplarına göre renklendir
            sample_size = min(5000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            
            fig = px.scatter(
                df_sample,
                x=feature,
                y=target_col,
                color='popularity_group',
                color_discrete_sequence=PROFESSIONAL_PALETTE,
                title=f"{feature} vs Popularity (r={corr_val:.3f}) - Gruplar",
                opacity=0.6
            )
            fig = apply_premium_layout(fig, f"{feature} vs Popularity (r={corr_val:.3f})")
            print(f"\nGörselleştirme: {feature} vs Popularity (Grouped)")
            save_figure(fig, f"phase3_scatter_{feature}_popularity_grouped")

print("\n" + "="*80)
print("PHASE 3 TAMAMLANDI")
print("="*80)
