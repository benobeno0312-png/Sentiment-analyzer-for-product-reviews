# Product Review Sentiment Analyzer

## Problem
Companies get thousands of reviews. Reading them manually is slow. This project classifies each review as
**Positive / Neutral / Negative**, detects which **aspects** (battery, camera, price...) customers talk about,
and shows trends in an interactive dashboard.

## Project structure
| File | Purpose |
|---|---|
| `data/reviews.csv` | Dataset (1,200 reviews: product, date, rating, text, sentiment) |
| `generate_dataset.py` | Script that creates the dataset (replace with a real one, see below) |
| `train_model.py` | Cleans text, trains TF-IDF + Logistic Regression / Naive Bayes, saves `model.joblib` |
| `dashboard.py` | Streamlit dashboard (KPIs, trends, aspects, comparison, live predictor) |
| `requirements.txt` | Dependencies |

## How to run
```bash
pip install -r requirements.txt
python generate_dataset.py     # optional, CSV already included
python train_model.py          # trains and saves model.joblib + confusion_matrix.png
streamlit run dashboard.py
```

## Pipeline
1. **Data**: review text, rating, product, date.
2. **Preprocessing**: lowercase, strip symbols, drop duplicates.
3. **Features**: TF-IDF with unigrams + bigrams.
4. **Models**: Logistic Regression (balanced class weights) vs Naive Bayes; best by macro F1.
5. **Evaluation**: precision, recall, F1, confusion matrix.
6. **Dashboard**: filters, sentiment split, monthly trends, aspect sentiment, top words, product comparison, live prediction.

## IMPORTANT: be honest about the dataset
The included dataset is **synthetic and template-based**, so the model scores ~100% F1. That is NOT a realistic
result. Before putting numbers on your resume:
1. Download a real dataset (Kaggle: "Amazon Fine Food Reviews", "Flipkart Product Reviews", "Amazon Cell Phones Reviews").
2. Keep the same columns: `review_id, product, date, rating, review_text, sentiment`
   (derive sentiment from rating: 1-2 Negative, 3 Neutral, 4-5 Positive).
3. Re-run `train_model.py`. Expect roughly 85-92% accuracy on real data.

## Upgrades to make it stand out
- Fine-tune **DistilBERT** with Hugging Face Transformers and compare against the TF-IDF baseline.
- Replace keyword aspect detection with true aspect-based sentiment (e.g. PyABSA).
- Add KeyBERT / LDA topic extraction and an LLM-generated pros/cons summary.
- Add Hinglish support and fake-review detection.
- Deploy on Streamlit Cloud or Hugging Face Spaces.

## Resume line (after using real data)
"Built a product review sentiment analyzer with TF-IDF + Logistic Regression (F1: <your score>) on <N> reviews,
with aspect-level insights and trend analysis in a Streamlit dashboard."

## Interview questions to prepare
- Why TF-IDF + LR as baseline? When would BERT be better?
- How did you handle class imbalance (class weights, F1 instead of accuracy)?
- Why is your synthetic-data score unrealistic, and how did you validate on real data?
- Failure cases: sarcasm, mixed-sentiment reviews, negation ("not good").
