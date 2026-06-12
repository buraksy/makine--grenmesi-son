"""
SPOTIFY API INTEGRATION - EXTERNAL DATA ENRICHMENT
===================================================
Spotify Web API ile artist ve track metadata ekleyerek model performansını iyileştirme.

Author: Model Expert Pipeline
Date: 19 Mayıs 2026
Expected Impact: +2-5% R² improvement
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ============================================================================
# SPOTIFY API CONFIGURATION
# ============================================================================

# GÜVENLİK NOTU: Production'da bu credentials environment variable'lardan alınmalı
SPOTIFY_CLIENT_ID = "4574d28451ce434f95a43d4dcdbb6597"
SPOTIFY_CLIENT_SECRET = "daab8caefe19482ca57aa7af2c3dc3b9"

# Rate limiting
MAX_REQUESTS_PER_SECOND = 25  # Spotify limit: 30, güvenli: 25
BATCH_SIZE = 50  # Spotify API batch limit
CHECKPOINT_INTERVAL = 1000  # Her 1000 track'te checkpoint

# Paths
DATA_PATH = "../data/model_ready"
OUTPUT_PATH = "../data/spotify_enriched"
CHECKPOINT_PATH = "../data/spotify_enriched/checkpoints"

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print("="*80)
print("SPOTIFY API INTEGRATION - EXTERNAL DATA ENRICHMENT")
print("="*80)

# ============================================================================
# PHASE 1: SPOTIFY API AUTHENTICATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: SPOTIFY API AUTHENTICATION")
print("="*80)

try:
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Test authentication
    test_result = sp.search(q="test", limit=1)
    print("✓ Spotify API authentication başarılı")
    print(f"✓ API client initialized")
    
except Exception as e:
    print(f"❌ Authentication hatası: {e}")
    print("\nLütfen Client ID ve Client Secret'ı kontrol edin.")
    exit(1)

# ============================================================================
# PHASE 2: DATASET LOADING
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: DATASET LOADING")
print("="*80)

# Load original dataset (track_id bilgisi için)
print("\nOriginal dataset yükleniyor...")
dataset = pd.read_csv("../data/raw/dataset.csv")
print(f"✓ Original dataset: {dataset.shape}")

# Load model-ready data
print("\nModel-ready data yükleniyor...")
X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv")
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv")

print(f"✓ X_train: {X_train.shape}")
print(f"✓ X_test: {X_test.shape}")

# Combine for processing
print("\nTrain ve test birleştiriliyor...")
X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)

print(f"✓ Combined dataset: {X_combined.shape}")

# Index'i kaldır (data leakage riski!)
if 'Unnamed: 0' in X_combined.columns:
    print("\n⚠️ 'Unnamed: 0' (index) sütunu kaldırılıyor (data leakage fix)...")
    X_combined = X_combined.drop('Unnamed: 0', axis=1)
    print("✓ Index sütunu kaldırıldı")

# ============================================================================
# PHASE 3: TRACK ID EXTRACTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: TRACK ID EXTRACTION STRATEGY")
print("="*80)

# Check if track_id exists in dataset
if 'track_id' in dataset.columns:
    print("✓ 'track_id' sütunu dataset'te mevcut")
    track_ids = dataset['track_id'].dropna().unique()
    print(f"✓ Unique track_id sayısı: {len(track_ids)}")
    
    # Dataset'teki row sayısı ile karşılaştır
    total_rows = len(X_combined)
    if len(track_ids) < total_rows * 0.5:
        print(f"⚠️ Track ID coverage düşük ({len(track_ids)}/{total_rows})")
        print("⚠️ Alternatif strateji gerekebilir")
else:
    print("❌ 'track_id' sütunu bulunamadı")
    print("⚠️ Track name + artist name ile arama yapılacak (daha yavaş)")
    track_ids = None

# Sample küçük bir subset ile test edelim (full run çok uzun sürer)
SAMPLE_SIZE = 5000  # İlk 5000 track ile test
print(f"\n⚠️ İLK TEST: İlk {SAMPLE_SIZE} track ile çalışılacak")
print("⚠️ Başarılıysa full dataset'e geçilir")

if track_ids is not None:
    track_ids_sample = track_ids[:SAMPLE_SIZE]
else:
    track_ids_sample = None

# ============================================================================
# PHASE 4: SPOTIFY API DATA EXTRACTION
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: SPOTIFY API DATA EXTRACTION")
print("="*80)

# Initialize results storage
spotify_data = []
failed_tracks = []

# Batch processing with Spotify API (50 tracks per request)
if track_ids_sample is not None:
    print(f"\nSpotify API'den {len(track_ids_sample)} track için veri çekiliyor...")
    print("⏱️ Batch API kullanılıyor (50 track/request)")
    print("⏱️ Tahmini süre: 2-3 dakika")
    
    batch_size = 50
    total_batches = (len(track_ids_sample) + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    # Process tracks in batches
    for batch_idx in tqdm(range(0, len(track_ids_sample), batch_size), 
                          desc="Processing track batches", 
                          total=total_batches):
        batch_ids = track_ids_sample[batch_idx:batch_idx + batch_size]
        
        try:
            # Get batch of tracks (1 request for up to 50 tracks)
            tracks_response = sp.tracks(batch_ids)
            tracks = tracks_response['tracks']
            
            # Extract unique artist IDs
            artist_ids = []
            track_artist_map = {}
            for track in tracks:
                if track and track['artists']:
                    artist_id = track['artists'][0]['id']
                    track_id = track['id']
                    track_artist_map[track_id] = artist_id
                    if artist_id not in artist_ids:
                        artist_ids.append(artist_id)
            
            # Get batch of artists (1 request for up to 50 artists)
            artists_data = {}
            if artist_ids:
                for artist_batch_idx in range(0, len(artist_ids), batch_size):
                    artist_batch = artist_ids[artist_batch_idx:artist_batch_idx + batch_size]
                    artists_response = sp.artists(artist_batch)
                    for artist in artists_response['artists']:
                        if artist:
                            artists_data[artist['id']] = artist
            
            # Combine track and artist data
            for track in tracks:
                if track is None:
                    continue
                    
                track_id = track['id']
                artist_id = track_artist_map.get(track_id)
                artist = artists_data.get(artist_id) if artist_id else None
                
                if artist:
                    features = {
                        'track_id': track_id,
                        'track_name': track['name'],
                        'artist_name': track['artists'][0]['name'],
                        'artist_id': artist_id,
                        'artist_followers': artist['followers']['total'],
                        'artist_popularity': artist['popularity'],
                        'track_popularity_api': track['popularity'],
                        'track_release_date': track['album']['release_date'],
                        'album_total_tracks': track['album']['total_tracks'],
                        'track_explicit': 1 if track['explicit'] else 0,
                        'artist_genres': ','.join(artist['genres'][:3]) if artist['genres'] else 'unknown',
                        'artist_genre_count': len(artist['genres'])
                    }
                    spotify_data.append(features)
                else:
                    failed_tracks.append(track_id)
        
        except Exception as e:
            print(f"\n⚠️ Batch {batch_idx//batch_size + 1} hatası: {str(e)}")
            failed_tracks.extend(batch_ids)
        
        # Rate limiting: 25 req/sec, her batch ~2 request (tracks + artists)
        # 50 batch = 100 request = 4 saniye min, güvenli olsun diye 0.1s sleep
        time.sleep(0.1)
        
        # Checkpoint save every 20 batches (~1000 tracks)
        if (batch_idx // batch_size + 1) % 20 == 0:
            checkpoint_df = pd.DataFrame(spotify_data)
            checkpoint_file = f"{CHECKPOINT_PATH}/spotify_data_checkpoint_{len(spotify_data)}.csv"
            checkpoint_df.to_csv(checkpoint_file, index=False)
            print(f"\n✓ Checkpoint kaydedildi: {len(spotify_data)} tracks")
    
    end_time = time.time()
    elapsed_time = round((end_time - start_time) / 60, 2)
    
    print(f"\n✓ Data extraction tamamlandı")
    print(f"✓ Başarılı: {len(spotify_data)}/{len(track_ids_sample)}")
    print(f"✓ Başarısız: {len(failed_tracks)}")
    print(f"✓ Süre: {elapsed_time} dakika")

else:
    print("❌ Track ID bulunamadığı için Spotify API extraction atlanıyor")
    print("⚠️ Alternatif: Track name + artist name ile search gerekli")
    exit(1)

# ============================================================================
# PHASE 5: DATA ENRICHMENT & FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: DATA ENRICHMENT & FEATURE ENGINEERING")
print("="*80)

# Convert to DataFrame
spotify_df = pd.DataFrame(spotify_data)
print(f"\n✓ Spotify data: {spotify_df.shape}")

# Feature engineering
print("\nYeni features oluşturuluyor...")

# 1. Artist followers (log transformation)
spotify_df['artist_followers_log'] = np.log1p(spotify_df['artist_followers'])

# 2. Artist popularity (0-1 normalized)
spotify_df['artist_popularity_normalized'] = spotify_df['artist_popularity'] / 100

# 3. Track age (days since release)
spotify_df['track_release_date'] = pd.to_datetime(spotify_df['track_release_date'], errors='coerce')
current_date = pd.Timestamp.now()
spotify_df['track_age_days'] = (current_date - spotify_df['track_release_date']).dt.days
spotify_df['track_age_months'] = spotify_df['track_age_days'] / 30

# 4. Is recent release (last 6 months)
spotify_df['is_recent_release'] = (spotify_df['track_age_days'] <= 180).astype(int)

# 5. Is single (album_total_tracks = 1)
spotify_df['is_single'] = (spotify_df['album_total_tracks'] == 1).astype(int)

# 6. Artist genre diversity
spotify_df['artist_genre_diversity'] = spotify_df['artist_genre_count'] / 10  # Normalize to 0-1

# 7. Artist follower per track (rough estimate of artist reach per release)
# This assumes average artist releases ~50 tracks
spotify_df['artist_follower_per_track'] = spotify_df['artist_followers'] / 50

print("✓ 7 yeni feature oluşturuldu:")
print("  1. artist_followers_log")
print("  2. artist_popularity_normalized")
print("  3. track_age_months")
print("  4. is_recent_release")
print("  5. is_single")
print("  6. artist_genre_diversity")
print("  7. artist_follower_per_track")

# ============================================================================
# PHASE 6: MERGE WITH ORIGINAL DATASET
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: MERGE WITH ORIGINAL DATASET")
print("="*80)

# Merge strategy: Dataset'teki track_id ile spotify_df'i eşleştir
# Ama burada bir sorun var: X_combined'da track_id yok
# Bu yüzden dataset → spotify_df → X_combined merge yapmalıyız

print("\nMerge stratejisi:")
print("1. Dataset + Spotify data → Enriched dataset")
print("2. Enriched dataset → X_combined'a feature transfer")

# Step 1: Dataset + Spotify data
if 'track_id' in dataset.columns:
    enriched_dataset = dataset.merge(
        spotify_df,
        on='track_id',
        how='left',
        suffixes=('', '_spotify')
    )
    print(f"✓ Enriched dataset: {enriched_dataset.shape}")
    
    # Check merge success
    merge_success_rate = (enriched_dataset['artist_followers'].notna().sum() / len(enriched_dataset)) * 100
    print(f"✓ Merge success rate: {merge_success_rate:.2f}%")
    
    # Save enriched dataset
    enriched_dataset.to_csv(f"{OUTPUT_PATH}/enriched_dataset.csv", index=False)
    print(f"✓ Enriched dataset kaydedildi: {OUTPUT_PATH}/enriched_dataset.csv")
    
else:
    print("❌ track_id bulunamadığı için merge yapılamıyor")
    exit(1)

# Step 2: Spotify features'ları X_combined'a ekle
# Bu kısım tricky çünkü X_combined'ın hangi row'unun hangi track'e karşılık geldiğini bilmiyoruz
# Bu yüzden index-based merge yapacağız (risky!)

print("\n⚠️ DİKKAT: X_combined'a feature ekleme risky (index-based merge)")
print("⚠️ Alternatif: Full pipeline yeniden çalıştırma önerilir")

# Select new features
new_feature_cols = [
    'artist_followers_log',
    'artist_popularity_normalized',
    'track_age_months',
    'is_recent_release',
    'is_single',
    'artist_genre_diversity'
]

# ============================================================================
# PHASE 7: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("PHASE 7: SAVE RESULTS")
print("="*80)

# Save spotify features separately
spotify_features = spotify_df[['track_id'] + new_feature_cols]
spotify_features.to_csv(f"{OUTPUT_PATH}/spotify_features.csv", index=False)
print(f"✓ Spotify features kaydedildi: {OUTPUT_PATH}/spotify_features.csv")

# Save summary report
summary = {
    'total_tracks_processed': len(track_ids_sample),
    'successful_extractions': len(spotify_data),
    'failed_extractions': len(failed_tracks),
    'success_rate': f"{(len(spotify_data)/len(track_ids_sample)*100):.2f}%",
    'new_features_created': len(new_feature_cols),
    'feature_names': new_feature_cols,
    'execution_time_minutes': elapsed_time,
    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(f"{OUTPUT_PATH}/spotify_integration_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary report kaydedildi: {OUTPUT_PATH}/spotify_integration_summary.json")

# ============================================================================
# PHASE 8: NEXT STEPS
# ============================================================================

print("\n" + "="*80)
print("PHASE 8: NEXT STEPS")
print("="*80)

print("\n✅ Spotify API integration tamamlandı!")
print("\n📋 Sonraki Adımlar:")
print("1. ⚠️ Enriched dataset'i kontrol et (merge quality)")
print("2. ⚠️ Full pipeline yeniden çalıştır (DataPrep → Model)")
print("3. ✅ Yeni features ile model retrain et")
print("4. 📊 Performance karşılaştırması yap")
print("5. 🎯 Beklenen: +2-5% R² improvement")

print("\n" + "="*80)
print("SPOTIFY API INTEGRATION TAMAMLANDI!")
print("="*80)
