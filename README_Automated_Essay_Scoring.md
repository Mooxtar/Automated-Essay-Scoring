# Automated Essay Scoring

Comparing classical ML, deep learning, and transformer-based approaches for automated essay scoring.

## Overview

This project builds and evaluates three different approaches to automatically score student essays on a scale of 1-6. The dataset contains 17,307 essays with human-assigned scores. The goal is to determine which modeling approach best captures writing quality.

## Approach

1. **Exploratory Data Analysis** - Analyzed score distributions, essay lengths, and text characteristics
2. **Classical ML Pipeline** - Feature engineering with NLP techniques + LightGBM
3. **Deep Learning** - BiLSTM with attention mechanism
4. **Transfer Learning** - Fine-tuned DistilBERT transformer model
5. **Comparison** - Evaluated all three approaches on the same test set

## Models

| Model | Description |
|-------|-------------|
| **LightGBM** | Gradient boosting on hand-crafted NLP features (word count, vocabulary richness, etc.) |
| **BiLSTM + Attention** | Bidirectional LSTM with attention layer for sequence modeling |
| **DistilBERT** | Fine-tuned pretrained transformer model |

## Tech Stack

- Python, NumPy, Pandas, Matplotlib
- scikit-learn, LightGBM, NLTK, WordCloud
- PyTorch, Hugging Face Transformers (DistilBERT)

## How to Run

Open and run `Automated_Essay_Scoring.ipynb` in Jupyter Notebook or Google Colab. The dataset is automatically downloaded from Google Drive. Note: the full pipeline takes approximately 4.5 hours to run (primarily due to transformer fine-tuning).
