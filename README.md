# Automated Essay Scoring

Comparing classical ML, deep learning, and transformer-based approaches for automated essay scoring.

## Overview

This project builds and evaluates four different models to automatically score student essays on a scale of 1-6. The goal is to determine whether complex deep learning architectures outperform classical ML with well-engineered features for this task.

## Dataset

- **17,307 essays** with human-assigned scores (1-6), no missing values
- **Imbalanced distribution**: scores 2, 3, and 4 made up ~86% of the data, while score 6 was extremely rare (<1%)
- **No prompt IDs provided** - topics were recovered using keyword matching and TF-IDF similarity
- Higher-scoring essays tended to be longer (median ~250 words for score 1 vs ~500+ for scores 5-6)

## Process

### 1. Exploratory Data Analysis
- Analyzed score distributions and identified class imbalance
- Examined essay length distributions (word count, sentence count, character count) across score levels
- Computed writing quality metrics: type-token ratio, average word length, vocabulary sophistication, punctuation usage
- Discovered that type-token ratio actually *decreased* for higher scores because longer essays naturally repeat more words
- Recovered essay topics using keyword matching and TF-IDF clustering, then analyzed how scoring patterns varied across topics

### 2. Data Splitting
- **Stratified 70/15/15 split** (train/validation/test) to preserve score proportions in every subset
- Test set was completely locked away until final evaluation
- Verified that proportions matched almost exactly across all three sets

### 3. Feature Engineering (for Classical ML)
Three groups of features were extracted:
- **Shallow features (15)**: word count, sentence length, vocabulary richness, punctuation usage, paragraph structure
- **TF-IDF word n-grams (1-2)**: captured actual vocabulary and phrases, with sublinear TF scaling
- **TF-IDF character n-grams (3-5)**: captured spelling patterns and morphological structure, more robust to typos
- Combined into ~20,000 features total using sparse matrix stacking

### 4. Model A: Classical ML

**Ridge Regression**:
- 5-fold stratified CV on training set to find best regularization alpha
- Trained final model with best alpha on full training set

**LightGBM**:
- 5-fold CV with early stopping to prevent overfitting
- Ensembled fold models by averaging predictions

**Feature Ablation**:
- Word TF-IDF alone was the strongest individual feature group
- Character n-grams added complementary value
- Shallow features alone were weaker but added structural signal that n-grams missed
- All combined gave the best result

**Interpretability**:
- Analyzed Ridge coefficients (positive = pushes score up, negative = pushes down)
- Examined LightGBM split-based feature importance

### 5. Model B: BiLSTM with Attention
- Used pretrained **GloVe 100d embeddings** (trained on 6B tokens from Wikipedia/Gigaword)
- Built vocabulary from training data, keeping words appearing 2+ times
- Tokenized and padded/truncated essays to 512 tokens
- Architecture:
  - Embedding layer (100d GloVe vectors)
  - Bidirectional LSTM (256d hidden state per direction)
  - Self-attention layer (learned which positions mattered most)
  - Dense head mapping to a single score
- Trained with MSE loss, Adam optimizer, learning rate scheduling, and early stopping
- Visualized attention weights to see which words the model focused on

### 6. Model C: DistilBERT
- Fine-tuned **DistilBERT** (40% smaller, 60% faster than BERT, retains 97% of understanding)
- Used HuggingFace tokenizer with max_length=512 (subword tokenization, attention masks)
- Applied **layer-wise learning rate decay**: lower layers (general linguistic knowledge) got smaller learning rates, higher layers and classification head got larger ones to avoid overwriting useful pretrained knowledge
- Used gradient-based token attribution to understand which tokens influenced predictions most
- Evaluated at different max token lengths to see how much of each essay DistilBERT actually needed

### 7. Data-Size Learning Curves
- Trained Ridge and LightGBM on increasing data fractions (10%, 25%, 50%, 75%, 100%)
- Measured how performance scaled with data size

## Results

### Test Set Performance

| Model | QWK | MAE | RMSE | Spearman rho |
|-------|-----|-----|------|-------------|
| **DistilBERT** | **0.7765** | 0.4682 | 0.5983 | 0.8223 |
| BiLSTM | 0.7716 | 0.4832 | 0.6208 | 0.8091 |
| Ridge | 0.7710 | 0.4786 | 0.6030 | 0.8097 |
| LightGBM | 0.7699 | 0.4522 | 0.5797 | 0.8288 |

### Statistical Significance (Paired Bootstrap, 10,000 iterations)
- DistilBERT vs Ridge: delta QWK = +0.0055, p = 0.225 (not significant)
- DistilBERT vs LightGBM: delta QWK = +0.0066, p = 0.162 (not significant)
- DistilBERT vs BiLSTM: delta QWK = +0.0049, p = 0.228 (not significant)
- **No statistically significant differences between any pair of models**

### Per-Score Analysis
- Rare classes (scores 1, 5, 6) were harder to predict correctly due to fewer training examples
- Models performed best on the majority classes (scores 2, 3, 4)

## Key Findings

- **All four models achieved similar QWK scores (~0.77)**, and no differences were statistically significant
- **Classical ML with good features is competitive with deep learning** for this task and dataset size - Ridge Regression with TF-IDF features matched DistilBERT's performance
- **LightGBM had the lowest MAE and RMSE** despite slightly lower QWK, meaning it made smaller errors on average
- **DistilBERT had the highest QWK** but the advantage was not statistically significant
- The feature ablation showed that **word-level TF-IDF was the most predictive single feature group**, confirming that the specific words students use strongly signal writing quality
- With only ~12K training essays, the deep learning models could not gain a meaningful advantage over well-engineered classical features

## Tech Stack

- Python, NumPy, Pandas, Matplotlib
- scikit-learn, LightGBM, NLTK, WordCloud
- PyTorch, Hugging Face Transformers (DistilBERT)
- GloVe embeddings (100d, 6B tokens)

## How to Run

Open and run `Automated_Essay_Scoring.ipynb` in Jupyter Notebook or Google Colab. The dataset is automatically downloaded from Google Drive. Note: the full pipeline takes approximately **4.5 hours** to run (primarily due to transformer fine-tuning).
