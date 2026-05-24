# Önerilen Modelleme Stratejisi


1. HEDEF: Şarkı Popülaritesi Tahmini (Regression - 0-100 arası popularity skoru)

2. ⚠️ KRİTİK UYARI: Audio Features ile Popularity Arasında Çok Zayıf İlişki!
   - En yüksek korelasyon: loudness (0.05) - bu çok düşük
   - Mevcut feature'larla R² < 0.10 beklenmeli
   - Model başarısı için external features (sanatçı, playlist, yayın tarihi) KRİTİK

3. ÖNERİLEN MODELLER:
   a) Baseline: Linear Regression (R² baseline oluştur)
   b) Regularized: Ridge, Lasso, ElasticNet (düşük korelasyonlar için)
   c) Tree-based: Random Forest Regressor, XGBoost Regressor, LightGBM
   d) Neural Network: MLP Regressor (polynomial relationships test için)
   e) Ensemble: Stacking/Blending (çok küçük iyileştirme sağlayabilir)

4. ÖNCELİKLİ VERİ HAZIRLIĞI:
   - ⚠️ KRİTİK: track_genre ONE-HOT veya TARGET ENCODING (EN GÜÇLÜ FEATURE!)
   - ID sütunlarını çıkar (track_id, artists, album_name, track_name)
   - explicit: Binary encoding (0/1)
   - Sayısal değişkenler: StandardScaler (linear models için) veya RobustScaler
   - Train-test split: Random 80-20
   - ⚠️ Harici feature eklenmesi şiddetle önerilir (artist followers, playlist count, release date)

5. DEĞERLENDİRME METRİKLERİ:
   - R² (Coefficient of Determination) - Model başarısı
   - RMSE (Root Mean Squared Error) - Ortalama hata
   - MAE (Mean Absolute Error) - Mutlak hata
   - MAPE (Mean Absolute Percentage Error) - Yüzdelik hata
   - Residual Plot (hata dağılımı kontrol)
   - ⚠️ Beklenen R²: 0.05-0.15 (çok düşük - audio features yetersiz)

6. FEATURE ENGINEERING FıRSATLARı (Öncelik Sırasıyla):
   a) ⚠️ KRİTİK: track_genre encoding (EN GÜÇLÜ FEATURE - mutlaka ekle)
   b) ⚠️ YÜKSEK: External features (artist takipçi sayısı, playlist sayısı, yayın tarihi)
   c) Düşük: Audio polynomials (loudness², danceability²) - zaten zayıf
   d) Düşük: Mood score (valence x energy) - popularity ile ilişki yok
   e) Düşük: Feature interactions - lineer ilişki zaten yok

7. CROSS-VALİDATİON:
   - K-Fold (k=5 veya k=10) - Stratified gerekmez
   - Regression için random fold assignment yeterli

8. ⚠️ BEKLENTİ YÖNETİMİ:
   - Mevcut audio features ile yüksek accuracy BEKLENMEMELİ
   - Model başarısı düşük olabilir (R² < 0.15)
   - Popülerlik, müzikal özelliklerden çok pazarlama/platform faktörlerine bağlı
   - İyileştirme için artist_name, playlist_count, release_date gibi harici data gerekli
