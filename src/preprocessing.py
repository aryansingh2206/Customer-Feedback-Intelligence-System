import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean raw review text.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join(
        word for word in text.split()
        if word not in STOP_WORDS
    )
    return text


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and prepare clean review column.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)
    df["clean_review"] = df["review_text"].apply(clean_text)
    return df
