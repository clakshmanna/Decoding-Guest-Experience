from django.shortcuts import render, redirect
from django.contrib import messages
import pandas as pd
from textblob import TextBlob
import os

# ---------- GLOBAL DATA STORAGE ----------
DATASET = None
REVIEW_COLUMN = None


# ---------- SENTIMENT FUNCTION ----------
def get_sentiment(text):
    text = str(text)  # convert to string to avoid errors
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return "happy"
    elif polarity < 0:
        return "not satisfied"
    else:
        return "neutral"


# ---------- SIMPLE ASPECT EXTRACTION ----------
ASPECT_KEYWORDS = {
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "quality": ["quality", "material", "durable", "build"],
    "delivery": ["delivery", "shipping", "packaging"],
    "performance": ["performance", "speed", "battery", "power"]
}


def extract_aspect(text):
    text = str(text).lower()  # convert to string & lowercase
    aspects = []

    if "price" in text:
        aspects.append("price")
    if "quality" in text:
        aspects.append("quality")
    if "delivery" in text:
        aspects.append("delivery")
    if "packaging" in text:
        aspects.append("packaging")

    return aspects[0] if aspects else "general"


# ---------------- HOME PAGE ----------------
def home(request):
    sentiment = None

    if request.method == "POST":
        review = request.POST.get("review_text")

        if review:
            sentiment = get_sentiment(review)

    return render(request, "home.html", {"sentiment": sentiment})


# ---------------- UPLOAD DATASET ----------------
def upload_dataset(request):
    global DATASET, REVIEW_COLUMN

    if request.method == "POST":
        file = request.FILES.get("csv_file")

        if file and file.name.endswith(".csv"):
            df = pd.read_csv(file)

            # Auto-detect review column
            possible_cols = ['review', 'reviews', 'text', 'comment', 'review_text', 'reviews.text']
            review_col = None
            for col in possible_cols:
                if col in df.columns:
                    review_col = col
                    break

            if not review_col:
                messages.error(request, "No review column detected. Please upload a proper CSV with reviews.")
                return redirect("upload_dataset")

            DATASET = df
            REVIEW_COLUMN = review_col
            messages.success(request, f"Dataset uploaded successfully! Detected review column: '{review_col}'")
            return redirect("dashboard")
        else:
            messages.error(request, "Please upload a valid CSV file.")

    return render(request, "upload.html")


# ---------------- DASHBOARD ----------------
def dashboard(request):
    global DATASET, REVIEW_COLUMN

    if DATASET is None:
        messages.error(request, "Upload dataset first!")
        return redirect("upload_dataset")

    df = DATASET.copy()
    review_col = REVIEW_COLUMN

    # Sentiment analysis
    df["sentiment"] = df[review_col].apply(get_sentiment)

    # Aspect extraction
    df["aspect"] = df[review_col].apply(extract_aspect)

    # Overall sentiment counts
    happy = (df["sentiment"] == "happy").sum()
    not_satisfied = (df["sentiment"] == "not satisfied").sum()
    neutral = (df["sentiment"] == "neutral").sum()
    total = len(df)

    happy_percent = (happy / total) * 100 if total else 0
    not_satisfied_percent = (not_satisfied / total) * 100 if total else 0
    neutral_percent = (neutral / total) * 100 if total else 0

    not_sat_start = happy_percent
    neutral_start = happy_percent + not_satisfied_percent

    # Aspect summary
    aspect_summary = {}
    for aspect in df["aspect"].unique():
        temp = df[df["aspect"] == aspect]
        aspect_summary[aspect] = {
            "happy": (temp["sentiment"] == "happy").sum(),
            "not satisfied": (temp["sentiment"] == "not satisfied").sum(),
            "neutral": (temp["sentiment"] == "neutral").sum(),
        }

    context = {
        "happy": happy,
        "not_satisfied": not_satisfied,
        "neutral": neutral,
        "happy_percent": happy_percent,
        "not_sat_start": not_sat_start,
        "neutral_start": neutral_start,
        "aspect_summary": aspect_summary,
    }

    return render(request, "dashboard.html", context)
