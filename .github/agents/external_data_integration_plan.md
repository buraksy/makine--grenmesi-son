# EXTERNAL DATA INTEGRATION PLAN

**Tarih:** 19 Mayıs 2026  
**Proje:** Spotify Music Popularity Prediction  
**Mevcut Dataset:** 114,000 şarkı × 132 feature  
**Mevcut Model:** Bagging Regressor (R² = 0.6299)

---

## 1. MEVCUT DURUM ANALİZİ

### 1.1 Var Olan Veriler
- **Audio Features:** acousticness, danceability, energy, valence, tempo, loudness, etc.
- **Genre Data:** 114 genre (one-hot encoded)
- **Popularity:** 0-100 skalasında hedef değişken
- **Engineered Features:** mood_score, electronic_score, log transformations, interaction terms

### 1.2 Eksik/Zenginleştirilebilir Alanlar
- ❌ **Artist metadata:** Takipçi sayısı, popülerlik, kariyer süresi
- ❌ **Track metadata:** Release date, album info, isrc codes
- ❌ **Social metrics:** Streaming counts, playlist placements
- ❌ **Temporal data:** Release timing, trend analysis
- ❌ **Genre enrichment:** Alt-genre bilgisi, genre evolution
- ❌ **Audio analysis:** Detailed section analysis, tempo variations

---

## 2. EXTERNAL DATA KAYNAKLARI

### 2.1 Spotify Web API (Öncelik: YÜKSEK)
**Endpoint'ler:**
```
- GET /artists/{id} → Artist popularity, followers, genres
- GET /audio-analysis/{id} → Detailed audio analysis
- GET /tracks/{id} → Track popularity, release date, explicit
- GET /albums/{id} → Album release date, label
```

**Eklenebilecek Features:**
- `artist_followers`: Artist'in takipçi sayısı (popularity'i tahmin için güçlü sinyal)
- `artist_popularity`: Artist'in genel popülerliği (0-100)
- `track_release_date`: Şarkının yayın tarihi (temporal features)
- `track_age_days`: Şarkının yaşı (bugünden itibaren)
- `album_total_tracks`: Albümdeki toplam şarkı sayısı
- `artist_genre_count`: Artist'in kaç genre'da olduğu
- `is_major_label`: Major label kontrolü (Universal, Sony, Warner)

**Avantajlar:**
- ✅ Official Spotify API
- ✅ High-quality metadata
- ✅ Rate limit: 30 req/sec

**Dezavantajlar:**
- ❌ API authentication required (Client ID/Secret)
- ❌ 114,000 track için çok sayıda request (~3.8k request)
- ❌ Rate limiting management gerekli

### 2.2 MusicBrainz API (Öncelik: ORTA)
**Endpoint'ler:**
```
- GET /ws/2/recording → Recording metadata
- GET /ws/2/artist → Artist metadata
```

**Eklenebilecek Features:**
- `artist_begin_year`: Sanatçının kariyerine başlama yılı
- `artist_country`: Sanatçının ülkesi
- `recording_language`: Şarkı dili

**Avantajlar:**
- ✅ Open-source, free
- ✅ Rich metadata

**Dezavantajlar:**
- ❌ Slower API
- ❌ Spotify ID matching gerekli

### 2.3 Last.fm API (Öncelik: DÜŞÜK)
**Endpoint'ler:**
```
- track.getInfo → Play counts, listener counts
- artist.getInfo → Artist tags, play counts
```

**Eklenebilecek Features:**
- `lastfm_playcount`: Last.fm'de çalma sayısı
- `lastfm_listeners`: Benzersiz dinleyici sayısı
- `lastfm_tags`: Community tags

**Avantajlar:**
- ✅ Social listening data
- ✅ Free API

**Dezavantajlar:**
- ❌ Coverage problemi (tüm Spotify şarkıları yok)
- ❌ Spotify ID → Last.fm matching gerekli

### 2.4 YouTube Music API (Öncelik: DÜŞÜK)
**Eklenebilecek Features:**
- `youtube_view_count`: YouTube'da görüntülenme sayısı
- `youtube_like_count`: Beğeni sayısı
- `youtube_comment_count`: Yorum sayısı

**Avantajlar:**
- ✅ Very strong popularity signal

**Dezavantajlar:**
- ❌ Unofficial API (ytmusicapi)
- ❌ Matching complexity
- ❌ Copyright issues

---

## 3. ÖNERILEN IMPLEMENTATION PLANI

### 3.1 Phase 1: Spotify Web API Integration (ÖNCELİK)

**Adımlar:**
1. **API Authentication Setup**
   - Spotify Developer Dashboard'dan Client ID/Secret al
   - OAuth 2.0 authentication flow implement et
   - Token refresh mekanizması

2. **Data Extraction Script**
   ```python
   # Pseudo-code
   import spotipy
   from spotipy.oauth2 import SpotifyClientCredentials
   
   # Batch processing (100 tracks/request)
   for batch in track_ids:
       - Get track info
       - Get artist info
       - Get audio analysis
       - Rate limit handling (sleep if needed)
   ```

3. **Feature Engineering**
   - `artist_followers_log`: Log transformation
   - `artist_popularity_normalized`: 0-1 scaling
   - `track_age_months`: Release date → months since release
   - `is_recent_release`: Boolean (son 6 ay)
   - `artist_follower_per_track`: followers / total tracks

4. **Data Merging**
   - Spotify ID ile mevcut dataset'e join
   - Missing value handling
   - Validation

**Tahmini Süre:** 3-4 saat (API requests + processing)

**Beklenen Impact:** 
- Model R² improvement: +3-5% (artist popularity güçlü sinyal)
- New features: 10-15 adet

### 3.2 Phase 2: Temporal Feature Engineering

**Zaten var olan release date ile:**
- `release_year`: Yıl extraction
- `release_month`: Ay (seasonal effects)
- `release_day_of_week`: Hafta günü (Cuma çıkışlar popüler)
- `is_summer_release`: Yaz aylarında mı çıktı
- `decades_category`: 2000s, 2010s, 2020s

**Tahmini Süre:** 30 dakika

**Beklenen Impact:** +1-2% R²

### 3.3 Phase 3: Genre Hierarchy Enrichment

**MusicBrainz veya manual mapping ile:**
- `genre_parent`: Ana genre kategorisi (Rock, Pop, Electronic, etc.)
- `genre_level`: Alt-genre depth
- `is_mainstream_genre`: Pop, Rock, Hip-hop, Electronic → True

**Tahmini Süre:** 1-2 saat

**Beklenen Impact:** +0.5-1% R²

---

## 4. FEATURE PRIORITY RANKING

### High Priority (Impact > %3)
1. **artist_followers** - Artist'in takipçi sayısı
2. **artist_popularity** - Artist'in Spotify popülerliği
3. **track_age_days** - Şarkının yaşı

### Medium Priority (Impact 1-3%)
4. **release_month** - Mevsimsel effects
5. **album_total_tracks** - Album vs single
6. **artist_genre_count** - Genre diversity
7. **is_major_label** - Label effect

### Low Priority (Impact < 1%)
8. **lastfm_playcount** - Social listening
9. **artist_country** - Geographic effect
10. **youtube_view_count** - Cross-platform popularity

---

## 5. RISKS & MITIGATION

### Risk 1: API Rate Limiting
**Mitigation:**
- Batch requests (100 tracks/request)
- Exponential backoff retry strategy
- Progress checkpoint save (her 1000 track)
- Multi-threading ile parallelization

### Risk 2: Missing Data
**Mitigation:**
- Imputation strategies (median/mode)
- Missingness indicator features
- Fallback to default values

### Risk 3: Data Staleness
**Mitigation:**
- Cache API responses
- Periodic refresh mechanism
- Timestamp tracking

### Risk 4: Overfitting
**Mitigation:**
- Cross-validation ile test
- Feature selection (RFE, SHAP importance)
- Regularization artır

---

## 6. SUCCESS CRITERIA

### Minimum Viable Product (MVP)
- ✅ Spotify Web API integration
- ✅ Artist popularity + followers features
- ✅ Release date features
- ✅ Model R² improvement: +2%

### Stretch Goals
- ✅ MusicBrainz metadata
- ✅ Last.fm social metrics
- ✅ Model R² improvement: +5%
- ✅ Ensemble with external features

---

## 7. TIMELINE & RESOURCE ESTIMATION

### Option A: Spotify Only (Önerilen)
- **API Setup:** 30 min
- **Data Extraction:** 2-3 hours (114k tracks)
- **Feature Engineering:** 1 hour
- **Model Retraining:** 30 min
- **Total:** 4-5 saat

### Option B: Multi-Source (Full)
- **API Setup:** 1 hour (Spotify + MusicBrainz + Last.fm)
- **Data Extraction:** 6-8 hours
- **Matching & Cleaning:** 2 hours
- **Feature Engineering:** 2 hours
- **Model Retraining:** 1 hour
- **Total:** 11-14 saat

---

## 8. IMPLEMENTATION DECISION

**Önerim:**
Option A (Spotify Only) ile başla:
- En güçlü sinyaller Spotify API'den gelir
- Hızlı implementation
- Low risk
- Belirgin improvement beklentisi

Eğer sonuç pozitifse → Option B'ye expand et

---

## 9. NEXT STEPS

1. **Karar:** Hangi option'ı implement edeceğiz?
2. **API Keys:** Spotify Developer hesabı var mı?
3. **Execution:** Script oluşturma ve çalıştırma
4. **Validation:** Model performance comparison
5. **Documentation:** Results reporting

---

**Hazırlayan:** GitHub Copilot  
**Son Güncelleme:** 19 Mayıs 2026
