from pathlib import Path
from typing import Tuple, List, Set
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def initialize_nltk():
    """Download required NLTK resources if not present"""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download(["punkt", "stopwords", "wordnet"], quiet=True)


def preprocess_text(
    text: str, lemmatizer: WordNetLemmatizer, stop_words: Set[str]
) -> str:
    """Process text: lowercase, tokenize, remove stopwords, lemmatize"""
    text = text.lower()
    tokens = word_tokenize(text)

    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    return " ".join(processed_tokens)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset file with text;emotion pairs"""
    data: List[dict] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text, emotion = line.strip().split(";")
            data.append({"text": text, "emotion": emotion})
    return pd.DataFrame(data)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Preprocess text data and encode emotions"""
    initialize_nltk()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df["text"] = df["text"].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

    le = LabelEncoder()
    df["emotion_encoded"] = le.fit_transform(df["emotion"])

    return df, le
