"""
PHASE 1: DATA OVERVIEW
Amaç: Veri setinin temel yapısını anlamak
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# Klasörlerin varlığını garantile
Path('../data/processed').mkdir(parents=True, exist_ok=True)
Path('../figures').mkdir(parents=True, exist_ok=True)
Path('../reports/csv').mkdir(parents=True, exist_ok=True)
Path('../reports/markdown').mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 1: VERİ SETİ GENEL BAKIŞ ANALİZİ")
print("="*80)

# Veri setini yükle
df = pd.read_csv('../data/raw/dataset.csv')

print("\n1. VERİ SETİ BOYUTU")
print("-"*80)
print(f"Toplam Satır Sayısı: {df.shape[0]:,}")
print(f"Toplam Sütun Sayısı: {df.shape[1]}")
print(f"Toplam Hücre Sayısı: {df.shape[0] * df.shape[1]:,}")

print("\n2. İLK 5 SATIR")
print("-"*80)
print(df.head())

print("\n3. VERİ TİPLERİ")
print("-"*80)
dtype_counts = df.dtypes.value_counts()
print(dtype_counts)
print("\nDetaylı Veri Tipleri:")
print(df.dtypes)

print("\n4. EKSİK VERİ İLK BAKIŞ")
print("-"*80)
missing_summary = pd.DataFrame({
    'Sütun': df.columns,
    'Eksik Değer Sayısı': df.isnull().sum().values,
    'Eksik Değer Oranı (%)': (df.isnull().sum().values / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Eksik Değer Sayısı'] > 0].sort_values('Eksik Değer Oranı (%)', ascending=False)

if len(missing_summary) > 0:
    print(missing_summary)
else:
    print("✓ Veri setinde eksik değer bulunmamaktadır.")

print("\n5. SAYISAL DEĞİŞKENLER ÖZET İSTATİSTİKLER")
print("-"*80)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Sayısal Değişken Sayısı: {len(numeric_cols)}")
print("\nSayısal değişkenler:")
print(numeric_cols)
print("\nÖzet İstatistikler:")
print(df[numeric_cols].describe())

print("\n6. KATEGORİK DEĞİŞKENLER")
print("-"*80)
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"Kategorik Değişken Sayısı: {len(categorical_cols)}")
print("\nKategorik değişkenler:")
print(categorical_cols)

if len(categorical_cols) > 0:
    print("\nKategorik Değişken Eşsiz Değer Sayıları:")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} eşsiz değer")

print("\n7. POTANSIYEL HEDEF DEĞİŞKEN ANALİZİ")
print("-"*80)
# Hedef değişken olabilecek sütunları kontrol et
potential_targets = []
for col in df.columns:
    if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
        potential_targets.append(col)

if len(potential_targets) > 0:
    print(f"Potansiyel hedef değişken(ler): {potential_targets}")
else:
    print("Açık bir hedef değişken tespit edilemedi.")
    print("En uygun hedef değişken adayları:")
    for col in categorical_cols[:5]:
        print(f"  - {col}: {df[col].nunique()} eşsiz kategori")

print("\n8. ÖZET RAPOR")
print("-"*80)
summary_dict = {
    'Metrik': [
        'Toplam Satır',
        'Toplam Sütun',
        'Sayısal Değişken',
        'Kategorik Değişken',
        'Eksik Veri Olan Sütun',
        'Toplam Eksik Hücre',
        'Genel Eksik Veri Oranı (%)'
    ],
    'Değer': [
        f"{df.shape[0]:,}",
        df.shape[1],
        len(numeric_cols),
        len(categorical_cols),
        len(missing_summary),
        df.isnull().sum().sum(),
        round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2)
    ]
}
summary_df = pd.DataFrame(summary_dict)
print(summary_df)

# CSV'ye kaydet
summary_df.to_csv('../reports/csv/phase1_data_overview_summary.csv', index=False)
print("\n✓ Özet rapor kaydedildi: ../reports/csv/phase1_data_overview_summary.csv")

print("\n" + "="*80)
print("PHASE 1 TAMAMLANDI")
print("="*80)
