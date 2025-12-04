# Music Recommendation System


01_preprocess_data.py
  input: raw CSV/JSON/Parquet
  output: preprocessed.parquet

02_split_data.py
  input: preprocessed.parquet
  output: train.parquet, test.parquet

03_feature_engineering.py
  input: train.parquet, test.parquet
  output: features_train.parquet, features_test.parquet

04_train_models.py
  input: features_train.parquet
  output: model_x.pkl, model_y.pkl, metrics.json

05_predict.py
  input: model.pkl, features_test.parquet
  output: predictions.parquet
  
A complete, production-ready music recommendation system with data preprocessing, multiple models, advanced features, and comprehensive evaluation.

## 🚀 Quick Start

```bash
# 1. Preprocess data (one-time setup, ~4 minutes)
python3 -m src.main

# 2. Train models (~25 minutes)
python3 -m src.train_test_split
python3 -m src.als_recommender --factors 64 --iterations 15

# 3. Generate recommendations
python3 examples/recommendation_demo.py --user-id 12345
```

## 📚 Documentation

**Start here** → [**Complete Guide**](README_COMPLETE_GUIDE.md) - Comprehensive walkthrough with detailed preprocessing explanation

**Other guides:**
- [Recommendation Scripts](README_RECOMMENDATION_SCRIPTS.md) - Individual script documentation
- [Advanced Features](README_ADVANCED_FEATURES.md) - Similar tracks, ranking, evaluation
- [Quick Start](QUICK_START.md) - 5-step quick start guide
- [Data Format](data/preprocessed/README.md) - Preprocessed data documentation

## 🎯 What's Included

### Data Preprocessing
✅ Clean 800M+ interactions → 206M aggregated events  
✅ Deduplicate entities (60% artist reduction)  
✅ Create canonical IDs and sparse matrix  
✅ Memory-efficient processing (no OOM)  
✅ **Detailed 10-step explanation** in Complete Guide

### Recommendation Models
✅ **Popularity Baseline** - Simple but effective  
✅ **ALS Collaborative Filtering** - Personalized recommendations  
✅ **Similar Tracks** - Item-to-item recommendations  
✅ **Re-ranking** - Diversity and novelty optimization

### Evaluation & Metrics
✅ **9 comprehensive metrics**: Precision, Recall, NDCG, Coverage, Diversity, Novelty, etc.  
✅ **JSON results** for easy analysis  
✅ **Model comparison** tools  
✅ **Interpretation guidelines**

### Production Ready
✅ Pre-computed indices for fast serving  
✅ Saved models and encoders  
✅ API-ready code examples  
✅ Complete documentation

## 📊 System Overview

```
Raw Data (2.3GB)
    ↓
Data Preprocessing (10 steps)
    ├─ Standardization & cleaning
    ├─ Deduplication (canonical IDs)
    ├─ Filtering & aggregation
    └─ Sparse matrix creation
    ↓
Processed Data (1.7GB)
    ├─ 206M interactions
    ├─ 805K tracks
    ├─ 1.37M users
    └─ 99.98% sparse
    ↓
Model Training
    ├─ Train/test split
    ├─ Popularity baseline
    ├─ ALS model (64 factors)
    └─ Similar tracks index
    ↓
Serving & Evaluation
    ├─ Personalized recommendations
    ├─ Item-to-item similarity
    ├─ Re-ranking (diversity)
    └─ 9 quality metrics
```

## 🔑 Key Features

### Memory Efficient
- Processes 800M interactions without OOM
- Lazy evaluation with Polars
- Streaming writes with `sink_parquet()`
- 99.98% sparse matrix (523MB vs 4.4TB dense)

### Data Quality
- Zero NULL values in joins
- Canonical IDs for all entities
- Validated track references
- Temporal train/test split

### Advanced Capabilities
- **Diversity**: Using `track_group_id` to avoid recommending multiple versions
- **Novelty**: Promote discovery of new artists/genres
- **Multi-objective**: Balance accuracy, diversity, and novelty
- **Evaluation**: Track 9 metrics with JSON output

## 📦 Project Structure

```
src/
├── main.py                   # Main preprocessing pipeline
├── preprocess_data.py        # Data cleaning & transformation
├── train_test_split.py       # Temporal validation split
├── popular_tracks.py         # Popularity baseline
├── als_recommender.py        # ALS collaborative filtering
├── similar_tracks.py         # Item-to-item similarity
├── ranking.py                # Re-ranking with diversity
├── evaluation.py             # Comprehensive metrics
└── evaluate_models.py        # Full evaluation pipeline

examples/
├── load_data_example.py      # How to load data
├── recommendation_demo.py    # Generate recommendations
└── quick_evaluation.py       # Quick metrics demo

data/
├── raw/                      # Downloaded raw data
└── preprocessed/             # Processed data & models
    ├── items.parquet
    ├── events.parquet
    ├── label_encoders.pkl
    ├── interaction_matrix.npz
    ├── train_matrix.npz
    ├── test_matrix.npz
    ├── als_model.pkl
    └── evaluation_results/
```

## 🎓 Learn More

### Detailed Data Preprocessing
See [Complete Guide - Data Preprocessing Section](README_COMPLETE_GUIDE.md#data-preprocessing-step-by-step) for:
- 10 detailed preprocessing steps
- Why each step is needed
- Code explanations
- Example transformations
- Memory optimization techniques

### Model Training & Evaluation
- [Recommendation Scripts Guide](README_RECOMMENDATION_SCRIPTS.md)
- [Advanced Features Guide](README_ADVANCED_FEATURES.md)

### Quick Examples

```python
# Load preprocessed data
from src.load_preprocessed import load_interaction_data, load_catalog

matrix, encoders = load_interaction_data()
catalog = load_catalog()

# Load trained model
from src.als_recommender import ALSRecommender

model = ALSRecommender.load('data/preprocessed/als_model.pkl')

# Get recommendations
recommendations = model.recommend(user_id=12345, user_items=matrix, n=10)

# Display with track names
for track_id, score in recommendations:
    track_info = catalog.filter(pl.col('track_id') == track_id)
    print(f"{track_info['track_clean'][0]}: {score:.4f}")
```

## 📈 Results

**Data Statistics:**
- Users: 1,372,771
- Tracks: 804,714
- Interactions: 205,866,117
- Sparsity: 99.98%
- Date range: 2022-01-01 to 2022-12-31

**Model Performance (sample):**
- Precision@10: 0.0053 (normal for implicit feedback)
- Recall@10: 0.0044
- Hit Rate@10: 0.045 (4.5% users get relevant recommendation)
- Diversity@10: 1.0 (100% unique songs)
- Novelty@10: 0.30 (moderate novelty)

## 🛠️ Requirements

```bash
pip install polars numpy scipy scikit-learn implicit requests python-dotenv
```

## 📝 Citation

This project implements techniques from:
- Hu et al., "Collaborative Filtering for Implicit Feedback Datasets" (2008)
- Koren et al., "Matrix Factorization Techniques for Recommender Systems" (2009)

## ✅ Status

**Complete & Production Ready**
- ✅ All data preprocessing working
- ✅ Multiple models trained
- ✅ Comprehensive evaluation
- ✅ Full documentation
- ✅ Example code provided
- ✅ Memory optimized
- ✅ No NULL values
- ✅ Tested & verified

---

**Get Started:** Read the [Complete Guide](README_COMPLETE_GUIDE.md) for detailed walkthroughs!
