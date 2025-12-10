# Music Recommendation System

ML pipeline for building and evaluating music recommendations.

# Bucket name
s3-student-mle-20250101-65b9b79fea

## Project Structure

mle-project-sprint-4/
├── .venv_recsys/                     # Virual environment
├── catboost_info/                    # CatBoost training metadata
├── config/
│   ├── .env                          # Environment variables
│   └── requirements.txt              # Python dependencies
├── data/
│   ├── raw/                          # Raw data from S3
│   │   ├── tracks.parquet
│   │   ├── catalog_names.parquet
│   │   └── interactions.parquet
│   └── preprocessed/                 # Cleaned & transformed data
│       ├── items.parquet
│       ├── tracks_catalog_clean.parquet
│       ├── events.parquet
│       ├── train_events.parquet
│       ├── test_events.parquet
│       ├── train_matrix.npz
│       ├── test_matrix.npz
│       └── train_test_split_info.pkl
├── logs/                             # JSON logs for each pipeline step
│   ├── logs_main.json
│   ├── logs_raw_data_loading.json
│   ├── logs_data_preprocessing.json
│   ├── logs_train_test_split.json
│   ├── logs_popularity_based_model.json
│   ├── logs_als_model.json
│   ├── logs_similarity_based_model.json
│   ├── logs_rec_ranking.json
│   └── logs_rec_evaluation.json
├── models/                           # Trained model artifacts
│   ├── als_model.pkl
│   ├── catboost_classifier.cbm
│   └── label_encoders.pkl
├── notebooks/                        # Jupyter notebooks for data and results exploration
│   ├── data_overview.ipynb           # Raw data overview
│   ├── eda.ipynb                     # Exploratory data analysis
│   └── results.ipynb                 # Results overview, determination of the best model
├── results/                          # Model outputs & evaluation results
│   ├── top_popular.parquet
│   ├── popularity_track_scores.parquet
│   ├── personal_als.parquet
│   ├── similar.parquet
│   ├── similar_tracks_index.pkl
│   ├── recommendations.parquet
│   ├── evaluation_popularity.json
│   ├── evaluation_als.json
│   ├── evaluation_ranked.json
│   └── models_comparison.parquet
├── src/                              # Source code
│   ├──_init_.py
│   ├── main.py                       # Pipeline entry point
│   ├── logging_set_up.py             # Logging configuration
│   ├── s3_loading.py                 # Saving to S3 configuration
│   ├── s3_testing_connection.py      # Testing connection to S3
│   ├── raw_data_loading.py           # Step 2: Download raw data
│   ├── data_preprocessing.py         # Step 3: Data preprocessing
│   ├── train_test_split.py           # Step 4: Train/test split
│   ├── popularity_based_model.py     # Step 5: Popularity model
│   ├── als_model.py                  # Step 6: ALS model
│   ├── similarity_based_model.py     # Step 7: Similarity model
│   ├── rec_ranking.py                # Step 8: CatBoost ranking
│   ├── rec_evaluation.py             # Step 9: Model evaluation
│   └── preprocessed_data_loading.py  # Preprocessed data loading
├── .gitignore                        
└── README.md                         # Short project overview


## Pipeline Steps

### Step 1: Load Environment Variables
Loads configuration from `.env` file including paths to data directories
and S3 credentials for data access.

### Step 2: Download Raw Data
Downloads raw datasets from S3 storage:
- tracks.parquet — track metadata (features, duration, etc.)
- catalog_names.parquet — artist/album names mapping
- interactions.parquet — user-track interaction events
Can be skipped with `--skip-download` flag if data already exists locally.

### Step 3: Preprocess Data
Cleans and transforms raw data:
- Explodes dataframes
- Cleans and deduplicates tracks, artists, albums, genres names
- Filters invalid/missing entries
- Adds `track_group_id` indicator of tracks relation (original track, its versions, remixes, covers have one unique`track_group_id`, but different `track_id`) 
- Encodes categorical features (user ids, track ids)
- Outputs: `items.parquet`, `tracks_catalog_clean.parquet`, `events.parquet`

### Step 4: Split Data into Train/Test Sets
Splits interaction data chronologically:
- Creates train/test event dataframes by splitting on specified date_threshold or as quantile 
- Builds sparse user-item matrices (`train_matrix.npz`, `test_matrix.npz`)
- Ensures no data leakage between train and test periods

### Step 5: Popularity-Based Recommendations
Builds a baseline model using track popularity:
- Ranks tracks by total interaction count
- - Uses all train data
- Generates global "most popular" recommendations
- Output: `top_popular.parquet`, `popularity_track_scores.parquet`

### Step 6: ALS Recommendations
Trains Alternating Least Squares (ALS) collaborative filtering model:
- Learns latent user and item factors from interaction matrix
- Uses all train data for training
- Generates personalized recommendations per user for all train users_ids 
- Output: `als_model.pkl`, `personal_als.parquet`

### Step 7: Similarity-Based Recommendations
Computes item-item similarity:
- Uses pretrained ALS model, builds similarity index for all track_ids
- Results will be used for online recommendations, thus excluded from following steps (ranking and evaluation)
- Output: `similar.parquet`, `similar_tracks_index.pkl`

### Step 8: Ranking (CatBoost)
Trains a CatBoost classifier to re-rank candidates:
- Computes track-specific custom features: 
genre popularity, artist popularity, track popularity (by number of ralated tracks)
- Combines custom features with features from popularity_based and ALS models
- Uses all train data for training
- Learns to predict user-item relevance
- Output: `catboost_classifier.cbm`, `recommendations.parquet`

### Step 9: Evaluate Models
Computes evaluation metrics for all models:
- Computes popularity_based, ALS model and ranked model recommendations for test user_ids
- Uses all test data for evaluation or sample test users
- Computes metrics (Precision@K, Recall@K, MAP, NDCG) of each model
- Compares popularity, ALS and ranked models
- Output: `evaluation_*.json`, `models_comparison.parquet`


## Usage

1. Create virtual environment
- install extension
```bash
sudo apt-get install python3.10-venv
```
- create .venv
```bash
python3 -m venv .venv_recsys
```
- run .venv
```bash
source .venv_recsys/bin/activate
```
- install packages
```bash
pip install -r config/requirements.txt
```

2. Run pipeline
Run full pipeline
```bash
python3 -m src.main
```
or

skip data download (if raw data already exists)
```bash
python3 -m src.main --skip-download
```

3. Check Data Overview, EDA or Results
Take a look at 
notebooks/data_overview.ipynb
notebooks/eda.ipynb
notebooks/results.ipynb
Manually choose kernel Python(.venv_recsys)
Run all

4. Check logs
Take a look at 
logs/logs_main.py # Main pipeline logs
and separate scripts log files 