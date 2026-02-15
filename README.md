# Music Recommendation System
End-to-end machine learning system that builds personalized music recommendations and serves them via a FastAPI microservices architecture for real-time recommendations.
Combines popularity-based, collaborative filtering (ALS), similarity-based, and CatBoost ranking models to deliver blended online/offline recommendations.
The solution delivers comparative relevance of recommendations (0.0158 Precision@5, 0.0127 Recall@5), ranking quality (0.0190 NDCG@5), discovery and variety (1.59 Novelty@5) of recommendations.


## Problem Statement
### Business Context
Inspired by a real-world task of Yandex.Music, the music streaming platform wants to improve users engagement and listening experience, thus needs to buld personalized track recommendations system. The system should handle both new users (cold start) and returning users with listening history and suggest the next track in users playlists.

### ML Objective
Using music streaming data, build a multi-stage recommendation pipeline that:
- Generates **offline recommendations** using popularity-based and ALS collaborative filtering models;
- Generates **online recommendations** using item-item similarity;
- **Re-ranks** candidates with a CatBoost classifier using custom features;
- **Blends** offline and online recommendations into a final list;
- **Serves** final recommendations via FastAPI microservices.
Metrics of the solution quality should be focused on relevance of recommendations, ranking quality, discovery, and variety of recommendations. Thus, metrics choosen: 
- precision@5 - captures how many of the 5 recommended tracks are actually relevant;
- recall@5 - captures of all tracks considered relevant for a user, how many show up in the top 5;
- ndcg@5 - adds position sensitivity and graded relevance;
- novelty@5 - captures how “unpopular” or previously unseen the tracks are;
- diversity@5 - measures dissimilarity within the 5 tracks.


## High-Level Solution Architecture
```
Raw Data (S3)
    │
    ▼
Data Preprocessing ──► Train/Test Split
    │                       │
    ▼                       ▼
┌──────────────────────────────────────────┐
│         Model Training (on train set)    │
│  ┌─────────────┐  ┌───────────────────┐  │
│  │ Popularity   │  │ ALS Collaborative │  │
│  │ Model        │  │ Filtering         │  │
│  └──────┬───────┘  └────────┬──────────┘  │
│         │    ┌──────────────┤             │
│         │    │              │             │
│         ▼    ▼              ▼             │
│  ┌─────────────┐  ┌───────────────────┐  │
│  │ CatBoost    │  │ Similarity-Based  │  │
│  │ Ranking     │  │ Model             │  │
│  └──────┬──────┘  └────────┬──────────┘  │
└─────────┼──────────────────┼─────────────┘
          │                  │
          ▼                  ▼
   Offline Recs         Online Recs
   (port 8001)         (port 8003)
          │                  │
          └──────┬───────────┘
                 ▼
          Final Blended Recs ◄── Events Service
           (port 8000)            (port 8002)
```


## Project Structure

```
music_recs/
├── .env                                   # Environment variables
├── .gitignore
├── requirements.txt                       # Python dependencies
├── README.md
├── data/
│   ├── raw/                               # Raw data from S3
│   │   ├── tracks.parquet
│   │   ├── catalog_names.parquet
│   │   └── interactions.parquet
│   └── preprocessed/                      # Cleaned & transformed data
│       ├── items.parquet
│       ├── tracks_catalog_clean.parquet
│       ├── events.parquet
│       ├── train_events.parquet
│       ├── test_events.parquet
│       ├── train_matrix.npz
│       └── test_matrix.npz
├── logs/                                  # JSON logs for each pipeline step
├── models/                                # Trained model artifacts
│   ├── als_model.pkl
│   ├── catboost_classifier.cbm
│   └── label_encoders.pkl
├── notebooks/                             # Jupyter notebooks
│   ├── data_overview.ipynb                # Raw data overview
│   ├── eda.ipynb                          # Exploratory data analysis
│   └── results.ipynb                      # Results & model comparison
├── results/                               # Model outputs & evaluation
│   ├── evaluation_als.json
│   ├── evaluation_popularity.json
│   ├── evaluation_ranked.json
│   ├── feature_importances.png
│   ├── models_comparison.parquet
│   ├── personal_als.parquet
│   ├── recommendations.parquet
│   ├── similar.parquet
│   └── top_popular.parquet
└── src/                                   # Source code
    ├── _init_.py
    ├── logging_setup.py                   # Logging configuration
    ├── s3_testing_connection.py           # S3 connection testing
    ├── microservice/                      # FastAPI microservices
    │   ├── _init_.py
    │   ├── events.py                      # User events (listening history)
    │   ├── final_recs.py                  # Blended recommendations
    │   ├── main_services.py               # Services launcher
    │   ├── offline_recs.py                # Offline recommendations
    │   ├── online_recs.py                 # Online (similarity) recommendations
    │   └── test_services.py               # Service tests
    └── recommendations/                   # ML pipeline modules
        ├── _init_.py
        ├── main_recs.py                   # Main pipeline entry point
        ├── s3_loading.py                  # S3 upload utilities
        ├── raw_data_loading.py            # Step 1-2: Load env & download data
        ├── data_preprocessing.py          # Step 3: Data preprocessing
        ├── train_test_split.py            # Step 4: Train/test split
        ├── popularity_based_model.py      # Step 5: Popularity model
        ├── als_model.py                   # Step 6: ALS model
        ├── similarity_based_model.py      # Step 7: Similarity model
        ├── rec_ranking.py                 # Step 8: CatBoost ranking
        ├── rec_evaluation.py              # Step 9: Model evaluation
        └── preprocessed_data_loading.py   # Data loading helpers
```


## Libraries
- **Polars** / **Pandas** — data manipulation and analysis
- **Implicit** — ALS collaborative filtering
- **CatBoost** — gradient boosting for candidate re-ranking
- **Scikit-learn** — evaluation metrics and utilities
- **SciPy** — sparse matrix operations
- **FastAPI** / **Uvicorn** — microservice API framework
- **python-dotenv** — environment variable management
- **Requests** — HTTP client for inter-service communication
- **Seaborn** — visualization
- **python-json-logger** — structured JSON logging
- **boto3** — AWS S3 integration


## Data
The dataset contains anonymized music streaming data of 1.4 million users and 1 million tracks and is used for educational purposes only.

### Download Links
Raw data is downloaded automatically from S3 during pipeline execution. 
Alternatively, data can be downloaded from the following sources and should be placed into `./data` directory:
- [data.csv](https://drive.google.com/file/d/1Uc2WbhW9U5-TtLr8X82QQxk36J9CMNCu/view?usp=sharing)


### Dataset Descriptions
| File | Description |
|------|-------------|
| `tracks.parquet` | Track metadata (features, duration, etc.) |
| `catalog_names.parquet` | Artist and album names mapping |
| `interactions.parquet` | User-track interaction events |


## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/iuliia-kuznetsova/music_recs.git
cd music_recs
```

### 2. Prepare Virtual Environment
Create virtual environment:
```bash
python3 -m venv venv_recsys
```

Activate virtual environment:
```bash
source venv_recsys/bin/activate  # Linux/Mac
```
or
```bash
.\venv_recsys\Scripts\activate   # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the recommendation pipeline
```bash
python3 -m src.recommendations.main_recs
```
or skip data download if raw data already exists locally:
```bash
python3 -m src.recommendations.main_recs --skip-download
```

### 4. Launch microservices
```bash
python3 -m src.microservice.main_services
```

### 5. Explore results
Open the Jupyter notebooks (select `.venv_recsys` as kernel):
- `notebooks/data_overview.ipynb` — raw data overview
- `notebooks/eda.ipynb` — exploratory data analysis
- `notebooks/results.ipynb` — model comparison and results

### 6. Stop services
```bash
# Stop microservices (Ctrl+C in the running terminal)
# Deactivate virtual environment
deactivate
```


## Approach

### Part 1: Recommendation Pipeline

1. **Load Environment Variables**:
   - Configuration loading from `.env` file (data paths, S3 credentials, model hyperparameters);

2. **Download Raw Data**:
   - Downloading of `tracks.parquet`, `catalog_names.parquet`, `interactions.parquet` from S3;

3. **Data Preprocessing**:
   - Catalogs exploding — expanding multi-valued `albums`, `artists`, `genres` arrays in each track into individual rows, producing a fully denormalized track–entity table;
   - Text standardization — NFKD normalization, diacritics stripping, lowercasing, punctuation removal, whitespace collapsing for all entity names;
   - Name cleaning — removing parenthetical/bracket content, `feat./ft./featuring` suffixes, version markers (`live`, `remix`, `acoustic`, `remastered YYYY`), then converting to title case;
   - Entity deduplication — grouping artists, albums, tracks, and genres by their standardized names (albums and tracks also scoped by canonical artist) to assign a single `canonical_id` per unique entity;
   - ID mapping — building `raw_id → canonical_id` dictionaries for tracks, artists, albums, genres, and applying them to the exploded table so every row uses canonical IDs;
   - Track grouping — normalizing track titles (stripping version/cover tags) and grouping by `(normalized_title, canonical_artist)` to assign a shared `track_group_id` to originals, remixes, covers, and live versions of the same song (used later for diversity);
   - Items table — joining canonical IDs with clean names and `track_group_id` into a single catalog (`items.parquet`), then building a compact `tracks_catalog_clean.parquet` lookup (one row per track);
   - Events building — filtering interactions by the 95th-percentile `track_seq` threshold, keeping only tracks present in the catalog (semi-join), and aggregating per `(user_id, track_id)` into `listen_count` and `last_listen` (`events.parquet`);
   - Label encoding — mapping user and track IDs to contiguous 0-based indices (`label_encoders.pkl`) for sparse-matrix construction and model training;

4. **Train/Test Split**:
   - Splitting interaction data chronologically by date threshold or quantile;
   - Building of sparse user-item matrices (`train_matrix.npz`, `test_matrix.npz`);

5. **Popularity-Based Model** (baseline):
   - Ranking tracks by total interaction count;
   - Generating of global "most popular" recommendations;
   - Output: `top_popular.parquet`, `popularity_track_scores.parquet`;

6. **ALS Collaborative Filtering**:
   - Learning of latent user and item factors from the interaction matrix;
   - Generating of personalized recommendations per user;
   - Output: `als_model.pkl`, `personal_als.parquet`;

7. **Similarity-Based Model**:
   - Building item-item similarity index by using pretrained ALS model;
   - Used for online recommendations (excluded from ranking and evaluation);
   - Output: `similar.parquet`, `similar_tracks_index.pkl`;

8. **CatBoost Ranking**:
   - Custom features computing: genre popularity, artist popularity, track group size;
   - Combining with popularity and ALS scores;
   - Training of a CatBoost classifier to predict user-item relevance;
   - Output: `catboost_classifier.cbm`, `recommendations.parquet`;

9. **Model Evaluation**:
   - Computation of Precision@K, Recall@K, MAP, NDCG for each model on test data;
   - Popularity, ALS, and ranked models comparison;
   - Output: `evaluation_*.json`, `models_comparison.parquet`.

### Part 2: Microservice Architecture

1. **Offline Recommendations** (port 8001) — serves precomputed personal recommendations;
2. **Events Service** (port 8002) — tracks user listening history in real time;
3. **Online Recommendations** (port 8003) — generates similarity-based recommendations for recently listened tracks;
4. **Final Recommendations** (port 8000) — blends offline and online recommendations, alternating between them.


## Service URLs
| Service | URL | Description |
|---------|-----|-------------|
| Final Recommendations | http://localhost:8000/docs | Blended recommendations API |
| Offline Recommendations | http://localhost:8001/docs | Precomputed personal recommendations |
| Events | http://localhost:8002/docs | User listening history |
| Online Recommendations | http://localhost:8003/docs | Similarity-based recommendations |

### Example API Calls
```bash
# Get final blended recommendations
curl -X POST "http://localhost:8000/recommendations?user_id=123&k=10"

# Get offline-only recommendations
curl -X POST "http://localhost:8001/get_recs?user_id=123&k=10"

# Add a listening event
curl -X POST "http://localhost:8002/put?user_id=123&track_id=456"

# Get user events
curl -X POST "http://localhost:8002/get?user_id=123&k=10"

# Get similar tracks
curl -X POST "http://localhost:8003/similar_tracks?track_id=456&k=10"
```


## Results

All three models were evaluated on 721 597 test users (out of 1 371 768 total) at k = 5:

| Metric | Popularity | ALS | CatBoost Ranked |
|--------|-----------|-----|-----------------|
| Precision@5 | 0.0064 | 0.0129 | **0.0158** |
| Recall@5 | 0.0029 | 0.0119 | **0.0127** |
| NDCG@5 | 0.0062 | 0.0157 | **0.0190** |
| Novelty@5 | 0.21 | **1.94** | 1.59 |
| Diversity@5 | 1.0 | 1.0 | 1.0 |

**CatBoost Ranked model is the best performer** across relevance metrics:
- **Precision@5 = 0.0158** — 2.5× better than popularity baseline, 1.2× better than ALS;
- **Recall@5 = 0.0127** — 4.4× better than popularity, 1.07× better than ALS;
- **NDCG@5 = 0.0190** — 3.1× better than popularity, 1.2× better than ALS — relevant tracks are placed higher in the recommendation list;
- **Novelty@5 = 1.59** — significantly more novel than popularity (0.21), slightly less novel than pure ALS (1.94);
- **Diversity@5 = 1.0** — identical across all models.

Feature importance analysis of the CatBoost ranker shows that popularity dynamics and ALS collaborative filtering scores are the dominant predictive signals, while catalog-level features (genre popularity, artist popularity, track group size) contribute less.

Detailed analysis and plots are available in `notebooks/results.ipynb`.


## Further Improvements
The project was carried out for educational purposes only. 
The solution performance can be futher improved by:
1. Alternative Models
- LightFM — hybrid matrix factorization model that can incorporate user/item side features (genre, artist, listening history metadata) into the factorization, potentially improving cold-start and overall precision;
- Neural Collaborative Filtering (NCF) — replaces the dot-product of ALS with a neural network to learn non-linear user-item interactions; can be implemented with PyTorch;
- BERT4Rec / SASRec — transformer-based sequential recommendation models that capture temporal patterns in listening sessions; well suited for the online recommendation stage;
- Two-Tower DNN — separate user and item encoder towers trained with contrastive loss; scalable for large catalogs and can leverage rich item features (audio embeddings, genre, artist);
- LambdaMART / LightGBM ranker — gradient-boosted listwise ranking as an alternative to pointwise CatBoost classification; directly optimizes NDCG;
- Multi-Armed Bandit (Epsilon-Greedy, Thompson Sampling) — for the online blending stage to adaptively balance exploration (novel tracks) vs exploitation (known preferences).
2. Feature & Data Enrichments
- Adding of user-level behavioral features to the ranking model (session length, skip rate, time-of-day, repeat listen ratio);
- Incorporation of audio content features (embeddings from pre-trained models like MusicNN or CLAP) for similarity and cold-start items;
- Using of track metadata features (BPM, energy, valence) from audio analysis APIs for richer item representations.
3. Infrastructure & Production
- Containerization of microservices with Docker;
- Monitoring and alerting for service health and model drift;
- Evaluation at larger k (10, 20);
- Adding of catalog coverage and serendipity metrics.


## Author
**Iuliia Kuznetsova**
February 2026
