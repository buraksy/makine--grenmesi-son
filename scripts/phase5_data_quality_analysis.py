"""
PHASE 5: DATA QUALITY & ANOMALY DETECTION
Amaç: Veri kalitesi risklerini sistematik biçimde belirlemek
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
print("PHASE 5: VERİ KALİTESİ & ANOMALY TESPİTİ")
print("="*80)

# Veri setini yükle
df = pd.read_csv('../data/raw/dataset.csv')

print("\n" + "="*80)
print("A. EKSİK VERİ ANALİZİ")
print("="*80)

# Eksik veri analizi
missing_count = df.isnull().sum()
missing_ratio = (missing_count / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    "Değişken": df.columns,
    "Eksik Değer Sayısı": missing_count.values,
    "Eksik Değer Oranı (%)": missing_ratio.values
})
missing_summary = missing_summary[missing_summary["Eksik Değer Sayısı"] > 0].sort_values("Eksik Değer Oranı (%)", ascending=False)

if len(missing_summary) > 0:
    print(f"\n⚠ Eksik Veri Tespit Edildi:")
    print(missing_summary)
    
    # Görselleştirme
    fig = px.bar(
        missing_summary,
        y='Değişken',
        x='Eksik Değer Oranı (%)',
        orientation='h',
        color='Eksik Değer Oranı (%)',
        color_continuous_scale=[(0, PROFESSIONAL_PALETTE[4]), (1, PROFESSIONAL_PALETTE[3])],
        title="Değişken Bazında Eksik Veri Oranları"
    )
    fig = apply_premium_layout(fig, "Değişken Bazında Eksik Veri Oranları")
    print(f"\nGörselleştirme: Eksik Veri Bar Chart")
    save_figure(fig, "phase5_missing_values")
    
    # Öneriler
    for _, row in missing_summary.iterrows():
        if row['Eksik Değer Oranı (%)'] > 30:
            priority = "Yüksek"
            recommendation = f"{row['Değişken']} için değişken çıkarma veya ileri imputasyon değerlendirilmelidir."
        elif row['Eksik Değer Oranı (%)'] > 10:
            priority = "Orta"
            recommendation = f"{row['Değişken']} için KNN, MICE veya domain temelli imputasyon değerlendirilmelidir."
        else:
            priority = "Düşük"
            recommendation = f"{row['Değişken']} için mean/median imputasyon veya silme yeterli olabilir."
        
        add_data_prep_recommendation(
            issue=f"Eksik veri: {row['Değişken']}",
            evidence=f"Eksik veri oranı %{row['Eksik Değer Oranı (%)']:.2f}.",
            recommendation=recommendation,
            priority=priority
        )
    
    missing_summary.to_csv('../reports/csv/phase5_missing_values_summary.csv', index=False)
    print(f"\n✓ Eksik veri raporu kaydedildi: ../reports/csv/phase5_missing_values_summary.csv")
else:
    print("\n✓ Veri setinde eksik değer bulunmamaktadır.")

print("\n" + "="*80)
print("B. DUPLICATE ROWS ANALİZİ")
print("="*80)

# Duplicate satır kontrolü
duplicate_count = df.duplicated().sum()
duplicate_ratio = (duplicate_count / len(df) * 100).round(2)

print(f"\nDuplicate Satır Sayısı: {duplicate_count}")
print(f"Duplicate Oranı: %{duplicate_ratio:.2f}")

if duplicate_count > 0:
    print(f"⚠ {duplicate_count} duplicate satır tespit edildi.")
    
    add_data_prep_recommendation(
        issue="Duplicate satırlar",
        evidence=f"{duplicate_count} duplicate satır tespit edildi (%{duplicate_ratio:.2f}).",
        recommendation="Data Prep Expert; duplicate satırların silinmesi veya korunma gerekçesinin değerlendirilmesi gerekir.",
        priority="Orta"
    )
else:
    print("✓ Duplicate satır bulunmamaktadır.")

print("\n" + "="*80)
print("C. OUTLIER ANALİZİ (IQR YÖNTEMİ)")
print("="*80)

# Sayısal değişkenler
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Unnamed: 0' in numeric_cols:
    numeric_cols.remove('Unnamed: 0')

outlier_records = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_ratio = (outlier_count / len(df) * 100).round(2)
    
    outlier_records.append({
        "Değişken": col,
        "Q1": round(q1, 4),
        "Q3": round(q3, 4),
        "IQR": round(iqr, 4),
        "Alt Sınır": round(lower_bound, 4),
        "Üst Sınır": round(upper_bound, 4),
        "Outlier Sayısı": outlier_count,
        "Outlier Oranı (%)": outlier_ratio
    })

outlier_summary = pd.DataFrame(outlier_records).sort_values("Outlier Oranı (%)", ascending=False)

print(f"\nOutlier Özeti:")
print(outlier_summary.head(10))

# Görselleştirme
fig = px.bar(
    outlier_summary.head(15),
    y='Değişken',
    x='Outlier Oranı (%)',
    orientation='h',
    color='Outlier Oranı (%)',
    color_continuous_scale=[(0, PROFESSIONAL_PALETTE[4]), (0.5, PROFESSIONAL_PALETTE[1]), (1, PROFESSIONAL_PALETTE[3])],
    title="Sayısal Değişkenlerde Outlier Oranları (İlk 15)"
)
fig = apply_premium_layout(fig, "Sayısal Değişkenlerde Outlier Oranları (İlk 15)")
print(f"\nGörselleştirme: Outlier Bar Chart")
save_figure(fig, "phase5_outlier_ratios")

# Yüksek outlier oranı için öneriler
high_outliers = outlier_summary[outlier_summary['Outlier Oranı (%)'] > 5]

if len(high_outliers) > 0:
    print(f"\n⚠ Yüksek Outlier Oranı Tespit Edildi (>%5):")
    print(high_outliers)
    
    for _, row in high_outliers.iterrows():
        add_data_prep_recommendation(
            issue=f"Yüksek outlier oranı: {row['Değişken']}",
            evidence=f"Outlier oranı %{row['Outlier Oranı (%)']:.2f} (IQR yöntemi).",
            recommendation="Data Prep Expert; winsorization, log dönüşümü, robust scaler veya outlier removal seçeneklerini değerlendirmelidir.",
            priority="Orta"
        )
else:
    print("\n✓ Kritik seviyede outlier oranı tespit edilmedi.")

outlier_summary.to_csv('../reports/csv/phase5_outlier_analysis.csv', index=False)
print(f"\n✓ Outlier analizi kaydedildi: ../reports/csv/phase5_outlier_analysis.csv")

print("\n" + "="*80)
print("D. SAYISAL DEĞİŞKEN MANTIK KONTROLÜ")
print("="*80)

# Negatif değer kontrolü (olmaması gereken değişkenler için)
non_negative_cols = ['popularity', 'duration_ms', 'tempo']

print(f"\nNegatif Değer Kontrolü:")
for col in non_negative_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"  ⚠ {col}: {negative_count} negatif değer")
            add_data_prep_recommendation(
                issue=f"Negatif değer: {col}",
                evidence=f"{negative_count} negatif değer tespit edildi.",
                recommendation="Data Prep Expert; bu değerlerin veri hatası olup olmadığını kontrol etmeli ve düzeltme stratejisi belirlemelidir.",
                priority="Yüksek"
            )
        else:
            print(f"  ✓ {col}: Negatif değer yok")

# Sıfır değer kontrolü
zero_checks = ['duration_ms']

print(f"\nSıfır Değer Kontrolü:")
for col in zero_checks:
    if col in df.columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            zero_ratio = (zero_count / len(df) * 100).round(2)
            print(f"  ⚠ {col}: {zero_count} sıfır değer (%{zero_ratio:.2f})")
            if zero_ratio > 1:
                add_data_prep_recommendation(
                    issue=f"Şüpheli sıfır değer: {col}",
                    evidence=f"{zero_count} sıfır değer tespit edildi (%{zero_ratio:.2f}).",
                    recommendation="Data Prep Expert; sıfır değerlerin mantıklı olup olmadığını kontrol etmeli, gerekirse imputasyon veya silme uygulamalıdır.",
                    priority="Orta"
                )
        else:
            print(f"  ✓ {col}: Sıfır değer yok")

print("\n" + "="*80)
print("E. KATEGORİK DEĞİŞKEN TUTARLILğI")
print("="*80)

# Kategorik değişkenler
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print(f"\nKategorik Değişken Tutarlılık Kontrolü:")

# ID sütunları hariç
non_id_categorical = [col for col in categorical_cols if 'id' not in col.lower() and 'name' not in col.lower()]

for col in non_id_categorical[:5]:  # İlk 5 kategorik değişken
    unique_count = df[col].nunique()
    print(f"\n  {col}:")
    print(f"    Eşsiz değer sayısı: {unique_count}")
    
    if unique_count < 200:  # Makul sayıda kategori varsa kontrol et
        # Boşluk veya büyük/küçük harf tutarsızlığı kontrolü
        sample_values = df[col].dropna().unique()[:10]
        print(f"    Örnek değerler: {sample_values}")
    else:
        print(f"    ⚠ Yüksek kardinalite ({unique_count} eşsiz değer)")

print("\n✓ Kategorik değişken tutarlılık kontrolü tamamlandı")

# Data Prep önerilerini kaydet
if len(data_prep_recommendations) > 0:
    recommendations_df = pd.DataFrame(data_prep_recommendations)
    recommendations_df.to_csv('../reports/csv/phase5_data_prep_recommendations.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ Data Prep Expert önerileri kaydedildi:")
    print(f"  ../reports/csv/phase5_data_prep_recommendations.csv")

print("\n" + "="*80)
print("PHASE 5 TAMAMLANDI")
print("="*80)
