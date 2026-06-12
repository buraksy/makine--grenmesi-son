# 🎵 Spotify Müzik Veri Seti - Keşifsel Veri Analizi Final Raporu

**Analiz Tarihi:** 18 Mayıs 2026 · **Güncelleme:** Haziran 2026  
**Veri Seti:** dataset.csv (Spotify Music Dataset)  
**Analiz Yöntemi:** 7 Aşamalı Agentik EDA Pipeline  
**EDA Hedefi:** `popularity` (0–100) sürekli değişken analizi  
**Modelleme Hedefi (güncel):** İkili sınıflandırma — 0–49 Düşük, 50–100 Yüksek  

> Bu rapordaki popularity analizleri (dağılım, korelasyon, genre) geçerlidir. Final model **sınıflandırma**dır; notebook'ta `popularity_class` oluşturulur. Güncel metrikler: `README.md` ve `MODEL_EXPERT_FINAL_HANDOFF.md`.

---

## 📋 YÖNETİCİ ÖZETİ

### Veri Seti Profili
- **Boyut:** 114,000 şarkı × 21 değişken (2.4 milyon veri hücresi)
- **Kapsam:** 114 farklı müzik türü, 9 audio feature, popularity (ham hedef: 0-100)
- **Sınıflandırma hedefi:** popularity < 50 → Düşük (0), ≥ 50 → Yüksek (1) — ~%74 / ~%26 dağılım
- **Kalite:** Yüksek - Eksik veri %0.0, duplicate satır yok
- **Potansiyel:** Popülerlik modellemesi için uygun veri seti; audio features ile zayıf ilişki

### Kritik Fırsatlar
1. **Büyük ve Temiz Veri Seti:** 114,000 kayıt, minimal veri kalitesi sorunu
2. **Dengeli Popularity Dağılımı:** Popularity ortalaması 33.2, medyan 35.0, skewness 0.046 (simetrik)
3. **Genre Bilgisi (EN GÜÇLÜ FEATURE):** track_genre, popularity'nin en güçlü açıklayıcısı

### Temel Riskler
1. **⚠️ KRİTİK:** Audio features ile popularity arasında ÇOK ZAYİF ilişki (korelasyon < 0.1)
2. **Sınıf dengesizliği:** Yüksek popüler şarkılar ~%26 — recall için `class_weight` gerekli
3. **External Features Gereksinimi:** Sanatçı ünü, playlist sayısı, yayın tarihi gibi harici veriler KRİTİK
4. **Multicollinearity:** Energy, tempo, danceability değişkenlerinde yüksek VIF (>10)

### Model Hazırlık Durumu
**Değerlendirme: ÖZELLİK HAZIR — Sınıflandırma ile Accuracy ~%85 (Random Forest, test seti)**

Veri seti modellemeye yakındır ancak şu ön işleme adımları gereklidir:
- ✅ Veri kalitesi mükemmel (eksik veri minimal)
- ⚠️ track_genre encoding KRİTİK (EN GÜÇLÜ FEATURE)
- ⚠️ Scaling gerekli (farklı ölçekler)
- ⚠️ ID sütunlarının çıkarılması zorunlu
- ⚠️ Harici feature eklenmesi şiddetle önerilir (artist takipçi sayısı, playlist count)

---

## 📊 PHASE 1: VERİ SETİ GENEL BAKIŞ

### Yapılan Analiz
Veri setinin temel yapısını anlamak için boyut, veri tipleri, eksik veri durumu ve değişken profilleri incelendi.

### 🧠 Koddan Elde Edilen Bulgular

**Veri Boyutu:**
- **114,000 satır** × **21 sütun** = 2,394,000 veri hücresi
- Büyük ölçekli veri seti - makine öğrenmesi için yeterli örnek sayısı

**Veri Tipleri:**
- **9 float değişken:** Audio features (danceability, energy, loudness vb.)
- **6 integer değişken:** Popularity, duration, key, mode, time_signature
- **5 string değişken:** Metadata (track_id, artists, album_name, track_name, track_genre)
- **1 boolean değişken:** explicit (açık içerik durumu)

**Eksik Veri Durumu:**
- **3 eksik değer** (artists, album_name, track_name sütunlarında birer tane)
- **Genel eksik veri oranı: %0.0** (ihmal edilebilir seviye)

**Hedef Değişken:**
- **popularity:** 0-100 arası sürekli değer (şarkı popülarite skoru)
- Regression problemi (sürekli değer tahmini)
- Ortalama: 33.2, Medyan: 35.0, Skewness: 0.046 (simetrik dağılım)

### 💡 Analitik Yorum

**Veri Büyüklüğü Perspektifi:**
114,000 şarkı kayıtlı veri setleri makine öğrenmesi için ideal büyüklüktedir. Bu boyut:
- Overfitting riskini azaltır
- Model genelleme gücünü artırır
- Regression modelleri için yeterli (ancak çok büyük değil)

**Veri Tipi Dengesi:**
Veri setinde sayısal (15), kategorik (5) ve boolean (1) değişkenler dengeli dağılmış. Audio features (9 değişken) sayısal olup doğrudan modellemeye uygun. Kategorik değişkenler encoding gerektirir.

**Veri Kalitesi:**
Eksik veri oranının %0.0 olması olağanüstü bir kalite işaretidir. Çoğu gerçek dünya veri setinde %5-20 arasında eksik veri görülür. Bu veri seti bu açıdan temiz ve kullanıma hazırdır.

**Hedef Değişken Yapısı:**
Popularity (0-100 aralsı sürekli değişken), dengeli ve simetrik dağılmış (skewness 0.046). Bu:
- Regression modelleme için uygun yapıda
- Model seçiminde ensemble veya deep learning yaklaşımlarını gerektirebilir
- Değerlendirme metriklerinde macro/weighted F1-score kullanımını zorunlu kılar

### ⚠️ Risk / Dikkat Edilmesi Gereken Nokta

1. **Yüksek Sınıf Sayısı:** 114 sınıf, model eğitimini zorlaştırabilir. Her sınıfın yeterli temsil edilmesi kritiktir.
2. **Metadata Sütunları:** track_id, artists, album_name, track_name sütunları identifier olup leakage riskine yol açabilir. Modellemeye dahil edilmemelidir.
3. **Kategorik Kardinalite:** artists (31,437 eşsiz) ve album_name (46,589 eşsiz) çok yüksek kardinaliteye sahip. Encoding stratejisi gerektirir.

### 🔁 Agent Etkileşim Notu

**Data Prep Expert için:**
- ID sütunları (track_id, artists, album_name, track_name) modelleme öncesi çıkarılmalı
- Eksik 3 satır silinebilir veya forward fill uygulanabilir (etki minimal)
- track_genre label encoding gerektirir (hedef değişken)
- explicit binary encoding gerektirir (True/False → 1/0)

---

## 📈 PHASE 2: TEK DEĞİŞKENLİ ANALİZ

### Yapılan Analiz
Her değişkenin dağılımı, istatistiksel özellikleri, çarpıklık, kurtosis ve outlier durumu bireysel olarak incelendi.

### 🧠 Koddan Elde Edilen Bulgular

#### Sayısal Değişkenler - Kritik Bulgular

**popularity (Popülerlik):**
- Ortalama: 33.2, Medyan: 35.0, Std Sapma: 22.3
- **Skewness: 0.046** (neredeyse simetrik, çok hafif sağa çarpık)
- **Outlier oranı: %0.00** (outlier yok denecek kadar az)
- **Yorum:** Popülerlik dengeli dağılmış. Hem popüler (yüksek skor) hem niche (düşük skor) şarkılar mevcut.

**duration_ms (Şarkı Süresi):**
- Ortalama: 228,029 ms (~3.8 dakika), Medyan: 212,906 ms (~3.5 dakika)
- **Skewness: 11.20** ⚠️ (ÇOK YÜKSEK sağa çarpıklık - uzun şarkılar var)
- **Outlier oranı: %4.93** (5,617 şarkı aykırı uzunlukta)
- **Yorum:** Şarkı süreleri genel olarak 3-4 dakika civarında ancak bazı şarkılar çok uzun (5.2 milyon ms = 87 dakika). Log dönüşümü gerekebilir.

**danceability (Dans Edilebilirlik: 0-1):**
- Ortalama: 0.567, Medyan: 0.580
- **Skewness: -0.40** (hafif sola çarpık - dans edilebilir şarkılar baskın)
- **Outlier oranı: %0.54** (minimal)
- **Yorum:** Çoğu şarkı orta-yüksek dans edilebilirlik seviyesinde. Dengeli dağılım.

**energy (Enerji: 0-1):**
- Ortalama: 0.641, Medyan: 0.685
- **Skewness: -0.54** (sola çarpık - enerjili şarkılar daha yaygın)
- **Outlier oranı: %0.00**
- **Yorum:** Şarkılar genel olarak yüksek enerjili. Düşük enerjili şarkılar azınlıkta.

**instrumentalness (Enstrümantal İçerik: 0-1):**
- Ortalama: 0.095, Medyan: 0.000
- **Skewness: 4.58** ⚠️ (ÇOK YÜKSEK sağa çarpıklık - çoğu şarkı vokal içeriyor)
- **Outlier oranı: %22.15** ⚠️ (EN YÜKSEK - 25,246 şarkı aykırı enstrümantal)
- **Yorum:** Çoğu şarkı vokal içeriyor (medyan=0). Tamamen enstrümantal şarkılar outlier olarak işaretleniyor.

**speechiness (Konuşma İçeriği: 0-1):**
- Ortalama: 0.087, Medyan: 0.046
- **Skewness: 4.43** (ÇOK YÜKSEK sağa çarpıklık - konuşma içeriği düşük)
- **Outlier oranı: %11.59** ⚠️ (13,211 şarkı - podcast/rap türleri olabilir)
- **Yorum:** Çoğu şarkıda konuşma içeriği minimal. Yüksek speechiness değerleri rap/hip-hop türlerinde beklenir.

**tempo (Tempo: BPM):**
- Ortalama: 122.1 BPM, Medyan: 122.0 BPM
- **Skewness: -0.05** (neredeyse simetrik)
- **Outlier oranı: %0.54** (minimal)
- **Yorum:** Tempo dengeli ve simetrik dağılmış. Çoğu şarkı 100-140 BPM aralığında.

#### Kategorik Değişkenler - Kritik Bulgular

**track_genre (Müzik Türü - HEDEF DEĞİŞKEN):**
- **114 eşsiz müzik türü**
- En sık görülen türler: pop, rock, hip-hop, electronic, jazz
- **Yorum:** Çok geniş tür çeşitliliği. Her türde ortalama ~1,000 şarkı (114,000 / 114).

**explicit (Açık İçerik):**
- **2 kategori:** False (açık içerik yok), True (açık içerik var)
- **Baskın kategori:** False (~%75-80 tahmini)
- **Yorum:** Çoğu şarkı açık içerik içermiyor. Binary encoding kolay.

**artists (Sanatçı):**
- **31,437 eşsiz sanatçı** ⚠️ (ÇOK YÜKSEK kardinalite)
- **Yorum:** Her sanatçı ortalama 3-4 şarkı. Target/frequency encoding gerekli veya modelden çıkarılmalı.

### 💡 Analitik Yorum

**Dağılım Dengesizlikleri:**
Bazı değişkenler (duration_ms, instrumentalness, speechiness) yüksek çarpıklık gösteriyor. Bu:
- Model performansını olumsuz etkileyebilir (özellikle linear modellerde)
- Log veya Box-Cox dönüşümü bu değişkenleri normalize edebilir
- Tree-based modeller (Random Forest, XGBoost) çarpıklıktan daha az etkilenir

**Outlier Yoğunluğu:**
Instrumentalness (%22) ve speechiness (%12) değişkenlerinde yüksek outlier oranı var. Ancak bu outlier'lar:
- Veri hatası değil, türe özgü özellikler olabilir (örn: enstrümantal caz, rap)
- Robust scaling veya winsorization uygulanabilir
- Tree-based modeller outlier'lara dayanıklıdır

**Feature Kalitesi:**
Audio features (danceability, energy, valence) dengeli ve modelleme için uygun dağılımlara sahip. Bu özellikler müzik türü sınıflandırması için güçlü ayırt ediciler olabilir.

### ⚠️ Risk / Dikkat Edilmesi Gereken Nokta

1. **Yüksek Çarpıklık:** duration_ms, instrumentalness, speechiness → Log dönüşümü değerlendirilmeli
2. **Yüksek Outlier Oranı:** instrumentalness, speechiness → Robust scaler veya outlier treatment
3. **Yüksek Kardinalite:** artists, album_name → Target encoding veya modelden çıkarma

### 🔁 Agent Etkileşim Notu

**Data Prep Expert için öneriler:**
- **duration_ms:** Log dönüşümü uygula (skewness=11.20)
- **instrumentalness, speechiness:** Robust scaler veya winsorization değerlendir
- **artists, album_name:** Yüksek kardinalite → Target encoding veya çıkar
- **explicit:** Binary encoding (True→1, False→0)

---

## 🔗 PHASE 3: İKİLİ DEĞİŞKEN ANALİZİ

### Yapılan Analiz
Değişkenler arasındaki ikili ilişkiler incelendi. Hedef değişken (popularity) ile sayısal ve kategorik değişkenler arasındaki korelasyonlar ve bağlantılar analiz edildi.

### 🧠 Koddan Elde Edilen Bulgular

**Hedef Değişken Dağılımı (popularity):**
- **Değer aralığı:** 0-100 (sürekli)
- **Ortalama:** 33.24, **Medyan:** 35.00
- **Skewness:** 0.046 (neredeyse simetrik - dengeli dağılım)
- **Popularity grupları:**
  - Çok Düşük (0-20): 29.98%
  - Düşük (20-40): 29.08%
  - Orta (40-60): 29.04%
  - Yüksek (60-80): 11.07%
  - Çok Yüksek (80-100): 0.84%

**⚠️ KRİTİK BULGU: Sayısal Değişkenler vs Popularity - ÇOK ZAYIF İLİŞKİ!**

**Audio Features ile Popularity Korelasyonları:**
- **loudness:** 0.0504 (en yüksek, ama çok zayıf)
- **speechiness:** -0.0449 (negatif, çok zayıf)
- **danceability:** 0.0354 (çok zayıf)
- **mode:** -0.0139 (çok zayıf)
- **duration_ms:** -0.0071 (çok zayıf)
- **key:** -0.0039 (yok denecek kadar zayıf)
- **energy:** 0.0011 (yok denecek kadar zayıf)

**Yorum:** TÜM audio features ile popularity arasındaki korelasyon <0.1 - bu, müzikal özelliklerinin popularity'yi neredeyse hiç açıklamadığını gösteriyor.

**Kategorik Değişkenler vs Popularity:**

**explicit vs popularity:**
- Explicit şarkılar: Ortalama popularity 36.5
- Non-explicit şarkılar: Ortalama popularity 32.9
- **Fark:** %11 (küçük ama gözlemlenebilir pozitif etki)

**track_genre vs popularity:**
- **En yüksek popularity türleri:**
  - pop-film: 59.3
  - k-pop: 56.9
  - chill: 53.7
  - sad: 52.4
- **En düşük popularity türleri:**
  - acoustic: 20-25
  - instrumental: 15-20

**Yorum:** Genre, audio features'tan ÇOK DAHA GÜÇLÜ popularity açıklayıcısı. Pop-film türü 2x daha popüler.

### 💡 Analitik Yorum

**⚠️ KRİTİK İÇGÖRÜ: Popülerlik, Müzikal Özelliklerle Açıklanamıyor**

Bu bulgular, şarkı popüleritesinin müzikal özelliklerden (danceability, energy, valence vb.) ziyade **external faktörlerle** belirlendiğini gösteriyor:
- **Sanatçı ünü ve takipçi sayısı**
- **Playlist yerleştirme stratejileri** (Spotify Discover Weekly, Today's Top Hits vb.)
- **Pazarlama ve promosyon bütçeleri**
- **Yayın tarihi ve trend zamanlaması**
- **Sosyal medya virüsleşme**

**Müzik Bilimi Perspektifi:**
Bulgular, "iyi müzik popüler olur" varsayımını çürütüyor. Loudness, energy, danceability gibi özelliklerin popularity üzerinde minimal etkisi var (<5% varyans açıklaması). Bu, müzik endüstrisinin pazarlama odaklı doğasını yansıtıyor.

**Modelleme Potansiyeli - DÜŞÜK RİSK:**
Mevcut audio features ile regression modeli **düşük performans** gösterecektir:
- **Beklenen R²:** 0.05-0.15 (çok düşük)
- **RMSE:** ~22 (popularity std sapması 22.3)
- **Model başarısı:** Minimal - audio features popularity'yi açıklamıyor

**EN GÜÇLÜ FEATURE: track_genre**
Genre bilgisi (pop-film 59.3 vs acoustic 20-25), audio features'tan ÇOK DAHA GÜÇLÜ popularity açıklayıcısı. Bu nedenle:
- track_genre mutlaka modele dahil edilmeli (One-Hot veya Target Encoding)
- Genre, popularity'nin en iyi proxy'si

### ⚠️ Risk / Dikkat Edilmesi Gereken Nokta

1. **⚠️ KRİTİK:** Mevcut feature'lar yetersiz - model başarısı düşük olacak
2. **External Features Gereksinimi:** Yüksek accuracy için artist_followers, playlist_count, release_date gibi harici veriler KRİTİK
3. **Beklenti Yönetimi:** R² < 0.15 beklenmeli - bu veri setiyle "iyi model" imkansız
4. **Feature Engineering Öncelikleri:** track_genre encoding > external features >> audio polynomials

### 🔁 Agent Etkileşim Notu

**Modeling Expert için:**
- ⚠️ Audio features popularity'yi açıklamıyor (korelasyon < 0.1)
- track_genre EN GÜÇLÜ FEATURE - mutlaka encode edilmeli
- Linear regression baseline yapılmalı ama R² < 0.10 beklenmeli
- Tree-based modeller (Random Forest, XGBoost) denenebilir ama yine düşük performans
- ⚠️ Harici data (artist, playlist, release date) olmadan yüksek accuracy imkansız
- Model değerlendirmesinde "başarılı model" beklentisi düşürülmeli

**Data Prep Expert için:**
- track_genre encoding KRİTİK (One-Hot veya Target Encoding)
- Harici feature kaynakları araştırılmalı (Spotify API, sanatçı metadata)
- Audio features scaling yapılmalı ama büyük etki yaratmayacak

---

## 🌐 PHASE 4: ÇOK DEĞİŞKENLİ ANALİZ

### Yapılan Analiz
Değişkenler arasındaki çoklu ilişkiler, korelasyonlar ve multicollinearity riski incelendi. VIF analizi yapıldı.

### 🧠 Koddan Elde Edilen Bulgular

**Yüksek Korelasyonlar (|r| > 0.7):**
1. **energy ↔ loudness:** r = 0.76 ⚠️
   - Enerjili şarkılar daha gürültülü
   - Beklenen pozitif ilişki (müzik teorisiyle uyumlu)

2. **energy ↔ acousticness:** r = -0.73 ⚠️
   - Enerjili şarkılar daha az akustik (elektronik)
   - Beklenen negatif ilişki (akustik vs elektronik zıtlığı)

**VIF (Variance Inflation Factor) Analizi:**
VIF > 10 → Yüksek multicollinearity riski

1. **tempo:** VIF = 15.22 ⚠️ (EN YÜKSEK)
2. **energy:** VIF = 15.02 ⚠️
3. **danceability:** VIF = 12.32 ⚠️

VIF 5-10 arası (Orta risk):
- **loudness:** VIF = 7.02
- **valence:** VIF = 6.52

VIF < 5 (Düşük risk):
- **acousticness, speechiness, instrumentalness, liveness:** VIF < 4

### 💡 Analitik Yorum

**Multicollinearity Etkisi:**
Tempo, energy ve danceability değişkenleri arasında yüksek VIF değerleri var. Bu:
- **Linear modellerde (Logistic Regression) sorun yaratabilir:** Katsayı tahminleri kararsız olabilir
- **Tree-based modellerde (Random Forest, XGBoost) problem değil:** Ağaç modelleri multicollinearity'e dayanıklıdır
- **Deep Learning'de dikkat gerektirir:** Regularization (L1/L2) kullanılabilir

**Çözüm Stratejileri:**
1. **Regularization:** Ridge veya Lasso regression kullan
2. **PCA:** Boyut indirgeme ile multicollinearity azalt
3. **Feature Selection:** Yüksek VIF'li değişkenlerden birini çıkar (örn: tempo veya energy)
4. **Tree-based Modeller Tercih Et:** Random Forest, XGBoost multicollinearity'den etkilenmez

**Energy-Loudness-Acousticness Üçgeni:**
Bu üç değişken arasındaki güçlü ilişki müzik karakterizasyonu açısından mantıklı:
- Enerjili şarkılar → Yüksek loudness + Düşük acousticness
- Akustik şarkılar → Düşük energy + Yüksek acousticness
- Bu ilişki feature engineering fırsatı sunuyor (composite features)

### ⚠️ Risk / Dikkat Edilmesi Gereken Nokta

1. **Linear Modellerde Risk:** Logistic Regression kullanılacaksa regularization zorunlu
2. **VIF > 15:** Tempo ve energy değişkenlerinde kritik multicollinearity
3. **Feature Selection Gerekebilir:** Yüksek VIF'li değişkenlerden biri çıkarılabilir

### 🔁 Agent Etkileşim Notu

**Data Prep Expert için:**
- Multicollinearity tespit edildi ancak Tree-based modellerde sorun yaratmaz
- Linear model kullanılacaksa Ridge/Lasso regularization gerekli
- PCA veya feature selection değerlendirilebilir (opsiyonel)

**Modeling Expert için:**
- Tree-based modeller tercih edilirse multicollinearity sorunu yok
- Linear model tercih edilirse regularization zorunlu
- Feature importance analizi hangi değişkenlerin gerçekten önemli olduğunu gösterecek

---

## 🔍 PHASE 5: VERİ KALİTESİ & ANOMALY TESPİTİ

### Yapılan Analiz
Eksik veri, duplicate satırlar, outlier'lar, negatif/sıfır değerler ve kategorik tutarsızlıklar sistematik olarak kontrol edildi.

### 🧠 Koddan Elde Edilen Bulgular

**Eksik Veri Analizi:**
- **3 eksik değer** (artists, album_name, track_name)
- **Eksik veri oranı:** %0.0 (ihmal edilebilir)
- **Yorum:** Veri kalitesi mükemmel seviyede

**Duplicate Satırlar:**
- **0 duplicate satır**
- **Yorum:** Veri seti temiz, her şarkı benzersiz

**Outlier Analizi (IQR Yöntemi):**

Yüksek outlier oranına sahip değişkenler (>%5):
1. **instrumentalness:** %22.15 outlier (25,246 şarkı) ⚠️
2. **speechiness:** %11.59 outlier (13,211 şarkı) ⚠️
3. **time_signature:** %10.66 outlier (12,157 şarkı) ⚠️
4. **liveness:** %7.58 outlier (8,642 şarkı) ⚠️
5. **loudness:** %5.41 outlier (6,173 şarkı) ⚠️

Düşük outlier oranına sahip değişkenler:
- **danceability, energy, valence, tempo, popularity:** <%1 outlier
- Bu değişkenler temiz ve dengeli dağılmış

**Negatif Değer Kontrolü:**
- **popularity, duration_ms, tempo:** Negatif değer yok ✓
- **Mantıksal tutarlılık:** Tüm değerler beklenen aralıklarda

**Sıfır Değer Kontrolü:**
- **duration_ms:** 1 şarkı süre=0 ⚠️ (veri hatası olabilir)
- **Yorum:** 1 satır silinebilir veya medyan ile doldurulabilir

### 💡 Analitik Yorum

**Outlier'ların Anlamı:**
Yüksek outlier oranları ilk bakışta endişe verici görünse de:
- **instrumentalness'te %22 outlier:** Tamamen enstrümantal şarkılar (jazz, classical) çoğunlukta vokalli şarkılara göre outlier sayılıyor. Bu **veri hatası değil, türe özgü özellik**.
- **speechiness'te %12 outlier:** Rap, hip-hop, podcast türlerinde konuşma içeriği yüksek. Yine **türe özgü özellik**.
- **time_signature'da %11 outlier:** 4/4'ten farklı ritimler (3/4, 5/4, 7/8) outlier sayılıyor. **Müzikal çeşitlilik**.

Bu outlier'lar:
- Silmek doğru değil (türe özgü bilgi kaybı)
- Robust scaling ile normalize edilebilir
- Tree-based modeller bu outlier'lara dayanıklıdır

**Veri Kalitesi Değerlendirmesi:**
- **Eksik veri:** Mükemmel (0.0%)
- **Duplicate:** Mükemmel (0 satır)
- **Mantıksal tutarlılık:** Mükemmel (1 sıfır değer hariç)
- **Genel kalite:** 10/10

Bu, Spotify'ın veri toplama ve temizleme süreçlerinin profesyonel olduğunu gösteriyor.

### ⚠️ Risk / Dikkat Edilmesi Gereken Nokta

1. **Outlier Yönetimi:** Türe özgü outlier'lar silinmemeli, robust scaling tercih edilmeli
2. **Duration=0 Satır:** 1 şarkı süre=0 → Bu satır silinebilir (1/114,000 etkisi minimal)
3. **Yanlış Outlier Yorumu:** Outlier'ları veri hatası olarak görmek hatalıdır

### 🔁 Agent Etkileşim Notu

**Data Prep Expert için:**
- **Outlier stratejisi:** Robust scaling öner (silme değil)
- **Duration=0 satır:** Sil veya medyan ile doldur
- **Eksik 3 satır:** Sil (etkisi minimal)
- **Genel öneri:** Veri temiz, minimal müdahale yeterli

---

## 💡 PHASE 6: İÇGÖRÜ ÜRETİMİ

### Yapılan Analiz
Önceki 5 fazdan elde edilen teknik bulgular anlamlı iş içgörülerine ve modelleme stratejilerine dönüştürüldü.

### 🧠 8 Kritik İçgörü

#### İçgörü 1: Büyük Ölçekli Müzik Veri Seti
**Kanıt:** 114,000 şarkı, 21 değişken, 9 audio feature  
**İş Değeri:** Geniş veri seti, makine öğrenmesi modelleri için yeterli örnek sayısı sağlıyor  
**Modelleme Etkisi:** Yüksek - Büyük veri seti overfitting riskini azaltır ve genelleme gücünü artırır

#### İçgörü 2: Dengeli Popularity Dağılımı - Regression Problemi
**Kanıt:** Popularity ortalaması 33.2, medyan 35.0, skewness 0.046 (neredeyse simetrik)  
**İş Değeri:** Hedef değişken dengeli ve simetrik dağılmış - hem popüler hem niche şarkılar dengeli temsil ediliyor  
**Modelleme Etkisi:** Orta - Dengeli dağılım regression modellemesi için uygun, ancak extreme değerler az (%0.84 çok yüksek popularity)

#### İçgörü 3: ⚠️ KRİTİK: Audio Features, Popularity'yi Açıklamıyor
**Kanıt:** En yüksek korelasyon loudness ile 0.0504. Tüm audio features ile popularity arası korelasyon <0.1  
**İş Değeri:** Şarkının popülerliği, müzikal özelliklerden ziyade external faktörlerle (sanatçı ünü, pazarlama, playlist yerleştirme) belirleniyor  
**Modelleme Etkisi:** ÇOK YÜKSEK RİSK - Mevcut feature'lar ile yüksek accuracy beklenmemeli. R² < 0.10 olası. Alternatif feature'lar (sanatçı, tür, yayın tarihi) gerekli

#### İçgörü 4: Multicollinearity Riski - Feature Engineering Fırsatı
**Kanıt:** 3 değişkende VIF > 10 (tempo: 15.2, energy: 15.0, danceability: 12.3)  
**İş Değeri:** Bazı müzik özellikleri birbirleriyle yüksek korelasyon gösteriyor - özellik seçimi gerekli  
**Modelleme Etkisi:** Orta - Regularization (Ridge/Lasso) veya PCA ile multicollinearity yönetilebilir

#### İçgörü 5: Yüksek Veri Kalitesi - Minimal Temizlik Gereksinimi
**Kanıt:** Eksik veri %0.0, duplicate satır yok, mantıksal tutarsızlık yok  
**İş Değeri:** Veri seti temiz ve kullanıma hazır - veri hazırlama maliyeti düşük  
**Modelleme Etkisi:** Yüksek - Hemen modelleme aşamasına geçilebilir, sadece feature engineering odaklı çalışma gerekli

#### İçgörü 6: Genre ile Popularity İlişkisi
**Kanıt:** Pop-film (59.3), k-pop (56.9), chill (53.7) en yüksek ortalama popularity. Acoustic (20-25) düşük.  
**İş Değeri:** Müzik türü, popularity'nin en güçlü açıklayıcı değişkeni olabilir (audio features'tan daha etkili)  
**Modelleme Etkisi:** Yüksek - track_genre kategorik encoding ile feature olarak kullanılmalı. One-hot veya target encoding ile model performansı artırılabilir

#### İçgörü 7: Energy-Loudness-Acousticness İlişkisi
**Kanıt:** energy-loudness korelasyonu 0.76, energy-acousticness korelasyonu -0.73  
**İş Değeri:** Enerjili şarkılar daha gürültülü ve daha az akustik - tutarlı müzik karakterizasyonu  
**Modelleme Etkisi:** Düşük - Bu ilişkiler popularity ile korelasyon göstermiyor, composite feature'lar minimal etki yapar

#### İçgörü 8: Explicit İçerik Etkisi
**Kanıt:** Explicit şarkıların ortalama popularity'si 36.5, explicit olmayan 32.9 (%11 fark)  
**İş Değeri:** Explicit içerik, popularity üzerinde küçük ama gözlemlenebilir pozitif etkiye sahip  
**Modelleme Etkisi:** Düşük-Orta - explicit feature olarak modele dahil edilmeli, ancak tek başına güçlü açıklayıcı değil

### 💡 Kritik Değişkenler (Regression - Popularity Tahmini)

**Hedef Değişken:**
- **popularity:** 0-100 arası popularity skoru

**ÇOK YÜKSEK Öneme Sahip Feature:**
- **track_genre:** EN GÜÇLÜ FEATURE - Tür bilgisi popularity'yi en iyi açıklıyor (kategorik encoding gerekli)

**Orta Öneme Sahip Feature:**
- **explicit:** Explicit içerik, popularity ile pozitif ilişkili (%11 fark)

**Düşük Öneme Sahip Features (Audio Features - Zayıf Korelasyon):**
- **loudness:** En yüksek korelasyon (0.05) - çok zayıf ama en iyi audio feature
- **danceability:** Zayıf korelasyon (0.04) - dans edilebilirlik popularity ile ilişkisiz
- **speechiness:** Zayıf negatif korelasyon (-0.04) - konuşma içeriği popularity düşürüyor
- **energy:** Korelasyon ~0 - enerji popularity'yi açıklamıyor
- **valence:** Korelasyon ~0 - pozitif ton popularity'yi açıklamıyor
- **acousticness:** Korelasyon ~0 - akustiklik popularity'yi açıklamıyor
- **tempo:** Korelasyon ~0 - tempo popularity'yi açıklamıyor

### 💡 Feature Engineering Fırsatları (Öncelik Sırasıyla)

1. **Genre One-Hot Encoding veya Target Encoding** (Öncelik: ÇOK YÜKSEK - KRİTİK)
   - track_genre kategorik değişkeni popularity'nin EN GÜÇLÜ açıklayıcısı - mutlaka encode edilmeli

2. **Artist/Album Features (Harici Veri)** (Öncelik: ÇOK YÜKSEK)
   - Sanatçı takipçi sayısı, önceki şarkı başarıları, playlist sayısı gibi external features eklenebilir
   - ⚠️ Audio features yetersiz - harici data KRİTİK

3. **Temporal Features (Yayın Tarihi)** (Öncelik: Yüksek)
   - Şarkı yayın tarihi, trend dönemleri, sezonsal etkiler popularity'yi etkileyebilir

4. **Audio Feature Polynomials** (Öncelik: Düşük)
   - loudness², danceability², feature interactions (her ne kadar zayıf olsalar da non-linear ilişki test edilebilir)

5. **Mood Score (valence x energy)** (Öncelik: Düşük)
   - Şarkının genel mood'unu temsil eden composite feature (her ne kadar popularity ile ilişki zayıf olsa da)

### 🔁 Agent Etkileşim Notu

**Data Prep Expert için Konsolide Öneriler:**
Toplam 29 öneri (3 yüksek, 22 orta, 4 düşük öncelikli)

**⚠️ KRİTİK Öncelikli (Yeni):**
- **track_genre encoding:** ONE-HOT veya TARGET ENCODING (EN GÜÇLÜ FEATURE - mutlaka yapılmalı)
- **External features:** Artist metadata, playlist count, release date (audio features yetersiz)

**Yüksek Öncelikli (3):**
1. Yüksek VIF: tempo, energy, danceability → Regularization veya PCA
2. Yüksek korelasyon: energy-loudness, energy-acousticness → Feature selection (ama popularity'ye etkisi minimal)
3. Kategorik encoding: explicit (binary), track_genre (one-hot/target)

**Orta Öncelikli (22):**
- Çarpıklık dönüşümleri (duration_ms, instrumentalness)
- Outlier treatment (robust scaling)
- Yüksek kardinalite yönetimi (artists, album_name - çıkarılmalı veya external metadata ile zenginleştirilmeli)

**Düşük Öncelikli (4):**
- Minimal eksik veri temizliği
- Audio feature polynomials (zaten zayıf korelasyon)

---

## ✅ PHASE 7: MODEL HAZIRLIK DEĞERLENDİRMESİ

### Yapılan Analiz
Verinin modelleme aşamasına hazır olup olmadığı 8 kritik kriterle değerlendirildi.

### 🧠 Model Hazırlık Kontrol Listesi

| Kriter | Durum | Açıklama |
|---|---|---|
| **1. Eksik Veri Yönetimi** | ✅ **HAZIR** | Eksik veri oranı %0.0, minimal müdahale yeterli |
| **2. Encoding Gereksinimi** | ⚠️ **İŞLEM GEREKLİ (KRİTİK)** | track_genre (EN GÜÇLÜ FEATURE) encoding gerektiriyor |
| **3. Scaling Gereksinimi** | ⚠️ **İŞLEM GEREKLİ** | Farklı ölçekler, StandardScaler/RobustScaler gerekli |
| **4. Outlier İşleme** | ⚠️ **İŞLEM ÖNERİLİR** | 3 değişkende >%10 outlier, robust scaling önerilir |
| **5. Target Distribution** | ✅ **DENGELİ** | Popularity simetrik dağılmış (skewness 0.046) |
| **6. Leakage Riski** | ✅ **DÜŞÜK RİSK** | ID sütunları çıkarılacak, leakage riski düşük |
| **7. Train-Test Split** | ✅ **RANDOM SPLIT UYGUN** | Regression problemi - stratified split gerekmez |
| **8. Multicollinearity** | ⚠️ **ORTA RİSK** | VIF>10 değişkenler var, regularization önerilir |

### 💡 Final Değerlendirme

**Hazırlık Durumu:**
- ✅ Hazır: 0/8 (%0.0)
- ⚠️ İşlem Gerekli/Kısmen Hazır: 3/8 (%37.5)

**Genel Değerlendirme: HAZIR DEĞİL (KIRMIZI) - Audio features yetersiz**

Veri seti modellemeye **yakındır** ancak ⚠️ **DÜŞÜK MODEL PERFORMANSI** beklenmeli:

**Zorunlu Adımlar:**
1. ⚠️ **KRİTİK:** track_genre ONE-HOT veya TARGET ENCODING (EN GÜÇLÜ FEATURE!)
2. ✅ ID sütunlarını çıkar (track_id, artists, album_name, track_name)
3. ✅ explicit: Binary encoding (True→1, False→0)
4. ✅ Sayısal değişkenler: StandardScaler veya RobustScaler
5. ✅ Train-test split: Random 80-20

**Şiddetle Önerilen:**
- ⚠️ **Harici feature eklenmesi:** artist_followers, playlist_count, release_date
  - Neden? Audio features popularity'yi açıklamıyor (korelasyon < 0.1)
  - Mevcut features ile R² < 0.15 beklenmeli

**Opsiyonel Adımlar:**
- duration_ms: Log dönüşümü (skewness yüksek)
- Outlier treatment: Robust scaling (instrumentalness, speechiness)
- Audio polynomials: loudness², danceability² (minimal etki beklenir)

### 💡 Önerilen Modelleme Stratejisi

**1. HEDEF:**
Şarkı Popülaritesi Tahmini (Regression - 0-100 arası popularity skoru)

**2. ⚠️ KRİTİK UYARI:**
Audio Features ile Popularity Arasında ÇOK ZAYIF İlişki!
- En yüksek korelasyon: loudness (0.05) - bu çok düşük
- Mevcut feature'larla R² < 0.10 beklenmeli
- Model başarısı için external features (sanatçı, playlist, yayın tarihi) KRİTİK

**3. ÖNERİLEN MODELLER:**

a) **Baseline:** Linear Regression
   - R² baseline oluştur (beklenen R²: 0.05-0.10)
   - Feature importance görmek için

b) **Regularized (ÖNERİLEN):** Ridge, Lasso, ElasticNet
   - Düşük korelasyonlar için uygun
   - Multicollinearity'e dayanıklı
   - Feature selection (Lasso) ile önemli features belirlenebilir

c) **Tree-based:** Random Forest Regressor, XGBoost Regressor, LightGBM
   - Multicollinearity'e dayanıklı
   - Outlier'lara dayanıklı
   - **track_genre encoding ile daha iyi performans** (kategorik feature'lar için güçlü)

d) **Neural Network:** MLP Regressor
   - Polynomial relationships test için
   - Beklenen iyileşme minimal

e) **Ensemble:** Stacking/Blending
   - Çok küçük iyileştirme sağlayabilir

**4. DEĞERLENDİRME METRİKLERİ:**
- **R² (Coefficient of Determination):** Model başarısı (beklenen: 0.05-0.15)
- **RMSE (Root Mean Squared Error):** Ortalama hata (beklenen: ~22, std sapma 22.3)
- **MAE (Mean Absolute Error):** Mutlak hata
- **MAPE (Mean Absolute Percentage Error):** Yüzdelik hata
- **Residual Plot:** Hata dağılımı kontrol

**5. CROSS-VALİDATİON:**
- K-Fold (k=5 veya k=10)
- Stratified split gerekmez (regression)

**6. BEKLENEN PERFORMANS:**
- **Baseline (Linear Regression):** R² = 0.05-0.10 (çok düşük)
- **Regularized (Ridge/Lasso):** R² = 0.08-0.12
- **Tree-based (XGBoost):** R² = 0.10-0.15 (track_genre encoding ile)
- **⚠️ Uyarı:** Tüm modeller DÜŞÜK performans gösterecek - audio features yetersiz

### ⚠️ Risk / Dikkat Edilmesi Gereken Nokta

1. **⚠️ KRİTİK:** Mevcut feature'lar yetersiz - model başarısı düşük olacak
2. **External Features Gereksinimi:** Yüksek accuracy için artist_followers, playlist_count, release_date gibi harici veriler KRİTİK
3. **Beklenti Yönetimi:** R² < 0.15 beklenmeli - bu veri setiyle "iyi model" imkansız
4. **Feature Engineering Öncelikleri:** track_genre encoding > external features >> audio polynomials
5. **Müzik Endüstrisi Realitesi:** Popülerlik, müzikal özelliklerden çok pazarlama/platform faktörlerine bağlı

### 🔁 Agent Etkileşim Notu

**Data Prep Expert için Final Checklist:**
1. ⚠️ **KRİTİK:** track_genre encoding (One-Hot veya Target Encoding)
2. ✅ ID sütunlarını çıkar
3. ✅ Encoding uygula (explicit binary)
4. ✅ Scaling uygula (StandardScaler veya RobustScaler)
5. ⚠️ Harici feature kaynakları araştır (Spotify API, artist metadata)

**Modeling Expert için:**
- ⚠️ Audio features popularity'yi açıklamıyor (korelasyon < 0.1)
- Baseline linear regression R² < 0.10 beklenmeli
- track_genre encoding KRİTİK - One-Hot veya Target Encoding dene
- Tree-based modeller kategorik features için daha iyi
- ⚠️ Harici data olmadan yüksek accuracy imkansız
- Model değerlendirmesinde "başarılı model" beklentisi düşürülmeli
- R² = 0.10-0.15 bile "iyi sonuç" sayılmalı (mevcut features ile)
3. ✅ Scaling uygula (StandardScaler veya RobustScaler)
4. ✅ Train-test stratified split (80-20)
5. ⚠️ Opsiyonel: Log dönüşümü, feature engineering

**Modeling Expert için:**
1. ✅ XGBoost veya LightGBM ile başla (tree-based optimal)
2. ✅ Stratified K-Fold cross-validation kullan
3. ✅ Macro F1-Score'u primary metrik olarak kullan
4. ✅ Confusion matrix ile karışan türleri analiz et
5. ✅ Feature importance analizi yap

---

## 📁 OLUŞTURULAN RAPORLAR VE GÖRSELLER

### CSV Raporları (reports/csv/)
1. `phase1_data_overview_summary.csv` - Veri seti genel özeti
2. `phase2_univariate_numeric_summary.csv` - Sayısal değişkenler özeti
3. `phase2_univariate_categorical_summary.csv` - Kategorik değişkenler özeti
4. `phase2_data_prep_recommendations.csv` - Phase 2 Data Prep önerileri
5. `phase3_bivariate_numeric_summary.csv` - İkili ilişkiler özeti
6. `phase4_correlation_matrix.csv` - Korelasyon matrisi
7. `phase4_vif_analysis.csv` - VIF analizi
8. `phase4_data_prep_recommendations.csv` - Phase 4 Data Prep önerileri
9. `phase5_missing_values_summary.csv` - Eksik veri özeti
10. `phase5_outlier_analysis.csv` - Outlier analizi
11. `phase5_data_prep_recommendations.csv` - Phase 5 Data Prep önerileri
12. `phase6_key_insights.csv` - 8 kritik içgörü
13. `phase6_critical_features.csv` - Kritik değişkenler listesi
14. `phase6_feature_engineering_opportunities.csv` - Feature engineering fırsatları
15. `phase6_all_data_prep_recommendations.csv` - Konsolide Data Prep önerileri (29 öneri)
16. `phase7_model_readiness_checklist.csv` - Model hazırlık kontrol listesi

### Grafikler (figures/)
**Phase 2 - Univariate:**
- Histogram ve boxplot grafikleri (14 sayısal değişken için)
- Kategori dağılım grafikleri (explicit, track_genre)

**Phase 3 - Bivariate:**
- track_genre dağılım grafiği (top 20 tür)
- Sayısal değişkenler vs track_genre boxplot (8 değişken)
- explicit vs track_genre bar chart
- danceability vs energy scatter plot

**Phase 4 - Multivariate:**
- Korelasyon matrisi heatmap (tüm sayısal değişkenler)
- Audio features korelasyon heatmap
- VIF analizi bar chart
- Scatter matrix (4 anahtar değişken)

**Phase 5 - Data Quality:**
- Eksik veri bar chart
- Outlier oranları bar chart

### Markdown Raporları (reports/markdown/)
1. `phase7_modeling_strategy.md` - Modelleme stratejisi dokümantasyonu
2. `EDA_FINAL_REPORT.md` - Bu kapsamlı final rapor

---

## 🎯 SONUÇ VE YOL HARİTASI

### Temel Başarılar
✅ 114,000 şarkı başarıyla analiz edildi  
✅ 21 değişken detaylı incelendi  
✅ 29 Data Prep önerisi oluşturuldu  
✅ 8 kritik içgörü üretildi  
✅ Model hazırlık yol haritası belirlendi  
⚠️ **KRİTİK BULGU:** Audio features popularity'yi açıklamıyor (korelasyon < 0.1)

### Veri Seti Güçlü Yanları
1. **Yüksek Veri Kalitesi:** Eksik veri minimal (%0.0), duplicate yok
2. **Dengeli Target Dağılımı:** Popularity simetrik (skewness 0.046), regression için uygun
3. **Büyük Veri:** 114,000 şarkı, overfitting riskini azaltıyor
4. **Genre Bilgisi (EN GÜÇLÜ FEATURE):** track_genre, popularity'nin en iyi açıklayıcısı

### Veri Seti Zayıf Yanları
1. **⚠️ KRİTİK:** Audio features ile popularity arasında ÇOK ZAYIF ilişki (korelasyon < 0.1)
2. **Düşük Model Performansı Beklentisi:** R² < 0.15 beklenmeli (mevcut features ile)
3. **External Features Gereksinimi:** Artist, playlist, release date gibi harici veriler KRİTİK
4. **Multicollinearity:** VIF>10 (tempo, energy, danceability) - ama popularity'ye etkisi minimal

### Bir Sonraki Adımlar

**1. Data Prep Expert Çalışması (Öncelik: ÇOK YÜKSEK)**
- ⚠️ **KRİTİK:** track_genre ONE-HOT veya TARGET ENCODING (EN GÜÇLÜ FEATURE!)
- ID sütunlarını çıkar
- Encoding uygula (explicit binary)
- Scaling uygula (StandardScaler veya RobustScaler)
- Train-test random split (80-20) - stratified gerekmez
- ⚠️ **Harici feature kaynakları araştır:** Spotify API (artist_followers, playlist_count, release_date)

**2. Feature Engineering (Öncelik: DÜŞÜK)**
- track_genre encoding (KRİTİK - yukarıda)
- Audio polynomials (minimal etki beklenir)
- Mood score (popularity ile ilişki yok)
- Feature interactions (lineer ilişki zaten yok)

**3. Modelleme (Öncelik: YÜKSEK - ama düşük performans beklentisiyle)**
- Baseline: Linear Regression (R² = 0.05-0.10 beklenmeli)
- Regularized: Ridge, Lasso, ElasticNet
- Ana Model: Random Forest Regressor, XGBoost Regressor (track_genre encoding ile)
- ⚠️ Beklenti Yönetimi: R² = 0.10-0.15 bile "iyi sonuç" sayılmalı

**4. Model Değerlendirme**
- K-Fold cross-validation (k=5 veya k=10) - stratified gerekmez
- R², RMSE, MAE primary metrikler
- Residual plot analizi
- Feature importance analizi (genre'nin gücünü görmek için)
- ⚠️ Düşük R² beklenmeli - bu veri setiyle "başarılı model" zor

**5. İyileştirme Stratejisi (Öncelik: ÇOK YÜKSEK)**
- Spotify API kullanarak artist metadata ekle (followers, verified, genre)
- Playlist count, play count, release date gibi temporal features ekle
- Social media virality indicators (varsa)
- ⚠️ Harici data olmadan yüksek accuracy imkansız

---

## 📞 İLETİŞİM VE EKİP KOORDİNASYONU

Bu EDA raporu **Data Prep Expert** ve **Modeling Expert** ekipleriyle paylaşılmalıdır.

**Data Prep Expert için:**
- 29 öneri hazır (3 yüksek, 22 orta, 4 düşük öncelikli)
- ⚠️ **KRİTİK:** track_genre encoding zorunlu (One-Hot veya Target)
- ⚠️ Harici feature kaynakları araştırılmalı (Spotify API)
- CSV raporları: `reports/csv/phase6_all_data_prep_recommendations.csv`

**Modeling Expert için:**
- ⚠️ **UYARI:** Audio features popularity'yi açıklamıyor (korelasyon < 0.1)
- Beklenen R²: 0.05-0.15 (çok düşük) - beklenti yönetimi kritik
- Model hazırlık kontrol listesi hazır
- Önerilen modeller: Ridge/Lasso (baseline), Random Forest/XGBoost (track_genre encoding ile)
- Stratejik doküman: `reports/markdown/phase7_modeling_strategy.md`

**⚠️ TÜM EKİPLERE KRİTİK NOT:**
- Popülerlik, müzikal özelliklerden çok pazarlama/platform faktörlerine bağlı
- Mevcut audio features ile yüksek accuracy beklenmemeli
- Harici data (artist, playlist, release date) olmadan "başarılı model" imkansız
- R² = 0.10-0.15 bile bu veri setiyle "iyi sonuç" olarak değerlendirilmeli

---

**Rapor Hazırlayan:** EDA Expert (Agentik EDA Uzmanı)  
**Metodoloji:** CRISP-DM / 7 Aşamalı EDA Pipeline  
**Tarih:** 18 Mayıs 2026  
**Durum:** ✅ TAMAMLANDI  
**Problem Tipi:** Regression (Popularity Tahmini: 0-100)  
**Hedef Değişken:** popularity

---

*Bu rapor, veri bilimi projelerinin CRISP-DM metodolojisinde "Data Understanding" aşamasını başarıyla tamamlamıştır. Bir sonraki aşama olan "Data Preparation" için hazır durumdadır. ⚠️ Ancak, mevcut feature'ların popularity'yi zayıf açıkladığı (R² < 0.15 beklenir) göz önünde bulundurulmalıdır.*
