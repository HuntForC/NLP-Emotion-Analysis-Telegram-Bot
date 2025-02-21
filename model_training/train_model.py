import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_data, preprocess_data

# File paths
MODEL_PATH = Path("model_training/model.pkl")
LABEL_ENCODER_PATH = Path("model_training/label_encoder.pkl")
TRAIN_DATA_PATH = Path("data/train.txt")
VAL_DATA_PATH = Path("data/val.txt")

# Model parameters
TFIDF_PARAMS = {"sublinear_tf": True, "max_features": 10000, "ngram_range": (1, 2)}

LOGISTIC_PARAMS = {
    "solver": "lbfgs",
    "C": 5.0,
    "penalty": "l2",
    "max_iter": 500,
    "class_weight": "balanced",
}

GRID_SEARCH_PARAMS = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [5000, 10000],
    "clf__C": [0.1, 1.0, 5.0],
    "clf__max_iter": [500, 1000, 2000],
    "clf__class_weight": ["balanced", None],
}


def load_and_preprocess(file_path: Path) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load and preprocess data"""
    df = load_data(file_path)
    return preprocess_data(df)


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, digits=3
        )
    )


def find_best_parameters(X_train: pd.DataFrame, y_train: pd.Series):
    """Find optimal model parameters using grid search"""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("clf", LogisticRegression(solver="lbfgs", penalty="l2")),
        ]
    )

    searcher = GridSearchCV(
        pipeline, GRID_SEARCH_PARAMS, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=2
    )

    searcher.fit(X_train, y_train)

    print(f"Best parameters: {searcher.best_params_}")
    print(f"Best validation score: {searcher.best_score_:.3f}")


def train_and_save_model():
    """Train model and save artifacts"""
    train_df, label_encoder = load_and_preprocess(TRAIN_DATA_PATH)
    val_df, _ = load_and_preprocess(VAL_DATA_PATH)

    X_train, y_train = train_df["text"], train_df["emotion_encoded"]
    X_val, y_val = val_df["text"], val_df["emotion_encoded"]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # evaluate_model(pipeline, X_val, y_val, label_encoder)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print("Model trained and saved!")


if __name__ == "__main__":
    train_and_save_model()
