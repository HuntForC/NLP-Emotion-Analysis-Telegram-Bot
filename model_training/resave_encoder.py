import joblib
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent.parent
LABEL_ENCODER_PATH = BASE_DIR / "model_training/label_encoder.pkl"


def resave_label_encoder():
    """Load and resave the label encoder with a more compatible protocol"""
    try:
        # Load with any protocol
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        # Resave with protocol 3 for better compatibility
        joblib.dump(label_encoder, LABEL_ENCODER_PATH, protocol=3)
        print(f"Successfully resaved label encoder at: {LABEL_ENCODER_PATH}")
    except Exception as e:
        print(f"Error processing label encoder: {e}")


if __name__ == "__main__":
    resave_label_encoder()
