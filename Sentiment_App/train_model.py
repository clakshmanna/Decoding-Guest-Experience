"""
train_model.py

Run from project root (or sentiment_app folder). Example:
python sentiment_app/ml/train_model.py /path/to/datafiniti_consumer_reviews.csv

Produces:
 - sentiment_app/ml/vectorizer.pkl
 - sentiment_app/ml/model.pkl
"""

import sys
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\S+|https\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def label_from_rating(r):
    try:
        r = float(r)
    except:
        return None
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

def main(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    # Datafiniti columns often: 'reviews.text', 'reviews.rating', 'id' or 'product_id', 'title'
    if 'reviews.text' in df.columns:
        df['text'] = df['reviews.text']
    elif 'review_text' in df.columns:
        df['text'] = df['review_text']
    elif 'text' in df.columns:
        df['text'] = df['text']
    else:
        raise ValueError("Cannot find text column. Provide Datafiniti CSV or ensure text column exists.")

    # Datafiniti may have a nested structure (multiple rows per product). For safety, explode lists if needed.
    df['text'] = df['text'].astype(str).map(clean_text)

    # find rating column
    rating_col = None
    for cand in ['reviews.rating', 'reviews.rating.1', 'rating', 'stars']:
        if cand in df.columns:
            rating_col = cand
            break

    if rating_col is None:
        print("No rating column found; trying to infer label by sentiment words (not recommended).")
        # fallback: create labels by heuristic (not ideal)
        df['label'] = df['text'].apply(lambda t: "Positive" if "good" in t or "great" in t else ("Negative" if "bad" in t or "terrible" in t else "Neutral"))
    else:
        df['label'] = df[rating_col].map(label_from_rating)

    df = df.dropna(subset=['text', 'label'])
    print("Total labeled rows:", len(df))

    # Optional: balance classes by downsampling majority
    counts = df['label'].value_counts()
    print("Class distribution before balancing:\n", counts.to_dict())
    min_count = counts.min()
    frames = []
    for lbl in counts.index:
        part = df[df['label'] == lbl]
        if len(part) > min_count:
            part = resample(part, replace=False, n_samples=min_count, random_state=42)
        frames.append(part)
    df_bal = pd.concat(frames).sample(frac=1, random_state=42)
    print("Balanced distribution:\n", df_bal['label'].value_counts().to_dict())

    X = df_bal['text'].values
    y = df_bal['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # create ml dir and dump
    ml_dir = os.path.join(os.path.dirname(__file__), '')
    target_dir = os.path.abspath(ml_dir)
    os.makedirs(target_dir, exist_ok=True)

    model_path = os.path.join(target_dir, 'model.pkl')
    vectorizer_path = os.path.join(target_dir, 'vectorizer.pkl')
    # pipeline includes both tfidf and clf — easiest to save whole pipeline:
    joblib.dump(pipeline, model_path)
    # if you prefer separate vectorizer & model:
    # joblib.dump(pipeline.named_steps['tfidf'], vectorizer_path)
    # joblib.dump(pipeline.named_steps['clf'], model_path)
    print("Saved pipeline to", model_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py /path/to/datafiniti.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
