# Amazon Reviews Rating Prediction (Deep Learning + NLP)

A deep learning project that predicts Amazon product ratings (1–5 stars) based on customer review text using Natural Language Processing (NLP) and multiple neural architectures including SVM, CNN, RNN, and LSTM.

---

## Overview

This project focuses on **multi-class text classification**, aiming to predict the **star rating** of Amazon product reviews within the **musical instruments** category.  
We used a combination of **Bag-of-Words**, **word embeddings**, and **deep learning models** to capture sentiment and rating cues from unstructured text data.

---

## Objectives

- Build a machine learning model to predict star ratings (1–5) from review text.
- Experiment with different neural architectures (Sequential NN, CNN, RNN, LSTM).
- Handle class imbalance and text preprocessing for effective training.
- Evaluate model performance and identify challenges in fine-grained sentiment prediction.

---

## Dataset

- **Source:** Amazon Musical Instruments Reviews Dataset  
- **Total Records:** 1,048,576 rows  
- **Features:** 12 (including `reviewText`, `overall`, `summary`, etc.)  
- **Target Variable:** `overall` (star rating: 1–5)  
- Due to compute constraints, a sampled subset of the data was used for model training and testing.

---

## Data Preprocessing

Key preprocessing steps:
- Text cleaning (punctuation, HTML tags, lowercase conversion)
- Stopword removal using NLTK
- Tokenization and vectorization (Bag-of-Words + Embeddings)
- Balancing skewed class distribution (resampling)
- Train-test split (80/20)

---

## Models Implemented

| Model | Features Used | Description | Accuracy |
|-------|----------------|--------------|-----------|
| **SVM (Baseline)** | Bag-of-Words | Linear classifier for textual data | ~48% |
| **Sequential Neural Network** | Bag-of-Words | Fully connected layers with dropout | ~56% |
| **CNN** | Embedding vectors | Captures local n-gram features | ~56% |
| **RNN** | Embedding vectors | Sequential learning for context | ~52% |
| **LSTM** | Embedding vectors | Long-term context tracking | ~54% |

**Best Performance:** CNN & Sequential NN — achieved ~56% accuracy  
**Key Challenge:** Class imbalance (ratings heavily skewed toward 4–5 stars)

---

## 🧪 Experimental Setup

- **Frameworks:** TensorFlow, Keras, scikit-learn, NLTK  
- **Language:** Python 3.9  
- **Notebook:** `AmaanAli-YashviPipaliya-ShatakshiSingh-SeanOnyskiw-SourceCode.ipynb`
- **Environment:** Google Colab / Jupyter Notebook  
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Confusion Matrix, Classification Report

---

## Results Summary

| Metric | CNN | LSTM | Sequential NN |
|---------|-----|------|----------------|
| Accuracy | 56% | 54% | 56% |
| Precision | 0.55 | 0.52 | 0.56 |
| Recall | 0.53 | 0.51 | 0.54 |
| F1-score | 0.54 | 0.51 | 0.55 |

- **Observation:** Models perform well on mid–high ratings (3–5 stars) but struggle with lower ones due to limited data.
- **Insight:** Future improvement can come from class-weighted loss, transformers (BERT), or larger embeddings.

---

## Requirements

To reproduce the results, install dependencies using:

```bash
pip install -r requirements.txt
