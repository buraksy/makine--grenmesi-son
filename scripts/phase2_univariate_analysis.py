"""
PHASE 2: UNIVARIATE ANALYSIS
Amaç: Her değişkenin tek başına davranışını anlamak
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

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
    except Exception as e:
        print(f"  ⚠ PNG kaydı yapılamadı (kaleido gerekebilir)")

print("="*80)
print("PHASE 2: TEK DEĞİŞKENLİ ANALİZ (UNIVARIATE ANALYSIS)")
print("="*80)

# Veri setini yükle
df = pd.read_csv('../data/raw/dataset.csv')

# Değişken tipleri
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Unnamed: 0 sütununu çıkar (index sütunu)
if 'Unnamed: 0' in numeric_cols:
    numeric_cols.remove('Unnamed: 0')

categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print("\n" + "="*80)
print("A. SAYISAL DEĞİŞKENLER ANALİZİ")
print("="*80)

univariate_numeric_results = []

for col in numeric_cols:
    print(f"\n{'='*80}")
    print(f"Değişken: {col}")
    print(f"{'='*80}")
    
    # Temel istatistikler
    mean_val = df[col].mean()
    median_val = df[col].median()
    std_val = df[col].std()
    min_val = df[col].min()
    max_val = df[col].max()
    
    # Skewness ve Kurtosis
    skewness = df[col].skew()
    kurtosis_val = df[col].kurtosis()
    
    # IQR ve Outlier Analizi
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_ratio = (outlier_count / len(df)) * 100
    
    print(f"\nTemel İstatistikler:")
    print(f"  Ortalama: {mean_val:.4f}")
    print(f"  Medyan: {median_val:.4f}")
    print(f"  Standart Sapma: {std_val:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"\nDağılım Özellikleri:")
    print(f"  Skewness: {skewness:.4f}")
    print(f"  Kurtosis: {kurtosis_val:.4f}")
    print(f"\nOutlier Analizi (IQR):")
    print(f"  Alt Sınır: {lower_bound:.4f}")
    print(f"  Üst Sınır: {upper_bound:.4f}")
    print(f"  Outlier Sayısı: {outlier_count}")
    print(f"  Outlier Oranı: {outlier_ratio:.2f}%")
    
    # Sonuçları kaydet
    univariate_numeric_results.append({
        'Değişken': col,
        'Ortalama': round(mean_val, 4),
        'Medyan': round(median_val, 4),
        'Std Sapma': round(std_val, 4),
        'Min': round(min_val, 4),
        'Max': round(max_val, 4),
        'Skewness': round(skewness, 4),
        'Kurtosis': round(kurtosis_val, 4),
        'Outlier Sayısı': outlier_count,
        'Outlier Oranı (%)': round(outlier_ratio, 2)
    })
    
    # Histogram
    fig = px.histogram(
        df, 
        x=col, 
        nbins=50,
        color_discrete_sequence=[PROFESSIONAL_PALETTE[0]],
        title=f"{col} - Histogram Dağılımı"
    )
    fig = apply_premium_layout(fig, f"{col} - Histogram Dağılımı")
    fig.update_traces(marker_line_width=1, marker_line_color="#FFFFFF")
    print(f"\nGörselleştirme: Histogram")
    save_figure(fig, f"phase2_histogram_{col}")
    
    # Boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df[col],
        name=col,
        marker_color=PROFESSIONAL_PALETTE[1],
        boxmean='sd'
    ))
    fig = apply_premium_layout(fig, f"{col} - Boxplot (Outlier Tespiti)")
    print(f"Görselleştirme: Boxplot")
    save_figure(fig, f"phase2_boxplot_{col}")
    
    # Data Prep önerileri
    if abs(skewness) > 1:
        add_data_prep_recommendation(
            issue=f"{col} değişkeninde yüksek çarpıklık",
            evidence=f"Skewness değeri {skewness:.4f} olarak hesaplandı (|skewness| > 1).",
            recommendation=f"Data Prep Expert; Log, Box-Cox veya Yeo-Johnson dönüşümü seçeneklerini değerlendirmelidir.",
            priority="Orta"
        )
    
    if outlier_ratio > 5:
        add_data_prep_recommendation(
            issue=f"{col} değişkeninde yüksek outlier oranı",
            evidence=f"IQR yöntemine göre outlier oranı %{outlier_ratio:.2f} olarak hesaplandı.",
            recommendation=f"Data Prep Expert; winsorization, robust scaler veya outlier removal seçeneklerini değerlendirmelidir.",
            priority="Orta"
        )

# Sayısal değişkenler özet tablosu
numeric_summary_df = pd.DataFrame(univariate_numeric_results)
numeric_summary_df.to_csv('../reports/csv/phase2_univariate_numeric_summary.csv', index=False)
print(f"\n{'='*80}")
print("✓ Sayısal değişkenler özet tablosu kaydedildi:")
print("  ../reports/csv/phase2_univariate_numeric_summary.csv")

print("\n" + "="*80)
print("B. KATEGORİK DEĞİŞKENLER ANALİZİ")
print("="*80)

univariate_categorical_results = []

for col in categorical_cols:
    print(f"\n{'='*80}")
    print(f"Değişken: {col}")
    print(f"{'='*80}")
    
    # Frekans tablosu
    freq_table = df[col].value_counts()
    freq_ratio = df[col].value_counts(normalize=True) * 100
    
    unique_count = df[col].nunique()
    most_common = freq_table.index[0] if len(freq_table) > 0 else None
    most_common_ratio = freq_ratio.iloc[0] if len(freq_ratio) > 0 else 0
    
    print(f"\nTemel İstatistikler:")
    print(f"  Eşsiz Kategori Sayısı: {unique_count}")
    print(f"  En Sık Görülen: {most_common}")
    print(f"  En Sık Görülen Oranı: {most_common_ratio:.2f}%")
    
    print(f"\nİlk 10 Kategori Frekansı:")
    print(freq_table.head(10))
    
    # Sonuçları kaydet
    univariate_categorical_results.append({
        'Değişken': col,
        'Eşsiz Kategori': unique_count,
        'En Sık Görülen': str(most_common),
        'Baskın Oran (%)': round(most_common_ratio, 2)
    })
    
    # Görselleştirme - yalnızca makul sayıda kategori varsa
    if unique_count <= 50:
        top_n = min(20, unique_count)
        plot_data = freq_ratio.head(top_n).reset_index()
        plot_data.columns = [col, 'Oran (%)']
        
        if unique_count <= 15:
            # Bar chart
            fig = px.bar(
                plot_data,
                x=col,
                y='Oran (%)',
                color=col,
                color_discrete_sequence=PROFESSIONAL_PALETTE,
                title=f"{col} - Kategori Dağılımı"
            )
            fig.update_layout(showlegend=False)
        else:
            # Horizontal bar chart
            fig = px.bar(
                plot_data,
                y=col,
                x='Oran (%)',
                orientation='h',
                color='Oran (%)',
                color_continuous_scale=[(0, PROFESSIONAL_PALETTE[0]), (1, PROFESSIONAL_PALETTE[1])],
                title=f"{col} - İlk {top_n} Kategori Dağılımı"
            )
        
        fig = apply_premium_layout(fig, f"{col} - Kategori Dağılımı")
        print(f"\nGörselleştirme: Bar Chart")
        save_figure(fig, f"phase2_bar_{col.replace(' ', '_')}")
    else:
        print(f"\n⚠ Çok yüksek kardinalite ({unique_count} eşsiz değer), görselleştirme atlandı.")
    
    # Data Prep önerileri
    if unique_count > 30 and col != 'track_id':
        add_data_prep_recommendation(
            issue=f"{col} değişkeninde yüksek kardinalite",
            evidence=f"Eşsiz kategori sayısı {unique_count} olarak hesaplandı.",
            recommendation=f"Data Prep Expert; rare label encoding, target encoding veya kategorilerin gruplandırılması seçeneklerini değerlendirmelidir.",
            priority="Orta"
        )
    
    if most_common_ratio > 70:
        add_data_prep_recommendation(
            issue=f"{col} değişkeninde dengesiz kategori dağılımı",
            evidence=f"Baskın kategori oranı %{most_common_ratio:.2f} olarak hesaplandı.",
            recommendation=f"Data Prep Expert; bu değişkenin modelleme gücünü artırmak için grup bazlı kodlama veya feature engineering seçeneklerini değerlendirmelidir.",
            priority="Düşük"
        )

# Kategorik değişkenler özet tablosu
categorical_summary_df = pd.DataFrame(univariate_categorical_results)
categorical_summary_df.to_csv('../reports/csv/phase2_univariate_categorical_summary.csv', index=False)
print(f"\n{'='*80}")
print("✓ Kategorik değişkenler özet tablosu kaydedildi:")
print("  ../reports/csv/phase2_univariate_categorical_summary.csv")

# Data Prep önerilerini kaydet
if len(data_prep_recommendations) > 0:
    recommendations_df = pd.DataFrame(data_prep_recommendations)
    recommendations_df.to_csv('../reports/csv/phase2_data_prep_recommendations.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ Data Prep Expert önerileri kaydedildi:")
    print(f"  ../reports/csv/phase2_data_prep_recommendations.csv")

print("\n" + "="*80)
print("PHASE 2 TAMAMLANDI")
print("="*80)
