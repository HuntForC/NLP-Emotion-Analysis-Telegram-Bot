from pathlib import Path
from typing import Tuple, List, Set
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from datasets import Dataset


def initialize_nltk():
    """Download required NLTK resources if not present"""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download(["punkt", "stopwords", "wordnet"], quiet=True)


def get_emotion_stopwords() -> Set[str]:
    """Get stopwords excluding emotion-related words"""
    all_stopwords = set(stopwords.words("english"))
    # Keep important emotional markers and negations
    words_to_keep = {
        "not",
        "no",
        "nor",
        "never",
        "cannot",
        "cant",
        "can't",
        "won't",
        "wont",
        "wouldn't",
        "wouldnt",
        "shouldn't",
        "shouldnt",
        "very",
        "really",
        "absolutely",
        "completely",
        "totally",
        "too",
        "so",
        "such",
    }
    return all_stopwords - words_to_keep


def preprocess_text(
    text: str, lemmatizer: WordNetLemmatizer, stop_words: Set[str]
) -> str:
    """Process text while preserving emotional markers"""

    # Convert to lowercase but preserve repeated letters (e.g., "nooo")
    original_text = text.lower()

    # Replace contractions
    text = re.sub(r"n't", " not", original_text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'d", " would", text)

    # Preserve emotional punctuation
    text = re.sub(r"[!]+", " ! ", text)  # Preserve exclamation marks
    text = re.sub(r"[?]+", " ? ", text)  # Preserve question marks
    text = re.sub(r"\.{2,}", " ... ", text)  # Preserve ellipsis

    # Remove other punctuation
    text = "".join(
        [
            char
            for char in text
            if char not in (set(string.punctuation) - {"!", "?", "."})
        ]
    )

    # Handle repeated characters (e.g., "nooo" -> "noo")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize, but keep negations and emotional intensifiers
    processed_tokens = []
    for i, token in enumerate(tokens):
        if token not in stop_words or token in {"not", "no", "never"}:
            # Special handling for negations
            if i > 0 and tokens[i - 1] in {"not", "no", "never"}:
                # Keep the original form for words after negations
                processed_tokens.append(token)
            else:
                processed_tokens.append(lemmatizer.lemmatize(token))

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
    stop_words = get_emotion_stopwords()  # Use emotion-aware stopwords

    df["text"] = df["text"].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))

    le = LabelEncoder()
    df["emotion_encoded"] = le.fit_transform(df["emotion"])

    return df, le


def load_and_preprocess(file_path: Path) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load and preprocess data"""
    df = load_data(file_path)
    return preprocess_data(df)


def convert_to_dataset(texts: list, labels: list) -> Dataset:
    """Convert data to HuggingFace Dataset format"""
    return Dataset.from_dict({"text": texts, "label": labels})


def prepare_datasets(
    train_path: Path, val_path: Path
) -> Tuple[Dataset, Dataset, LabelEncoder]:
    """Prepare training and validation datasets"""
    # Load and preprocess data
    train_df, label_encoder = load_and_preprocess(train_path)
    val_df, _ = load_and_preprocess(val_path)

    # Convert to datasets
    train_dataset = convert_to_dataset(
        train_df["text"].tolist(), train_df["emotion_encoded"].tolist()
    )
    val_dataset = convert_to_dataset(
        val_df["text"].tolist(), val_df["emotion_encoded"].tolist()
    )

    return train_dataset, val_dataset, label_encoder
