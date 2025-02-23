import os
import sys
import joblib
import logging
import torch
import torch.nn.functional as F
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update, constants
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
)
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from model_training.data_preprocessing import preprocess_text, initialize_nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class Config:
    """Bot configuration settings"""

    MODEL_DIR = Path("model_training")
    MODEL_PATH = MODEL_DIR / "emotion_bert"
    LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


def load_model_artifacts():
    """Load the trained model and label encoder"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
        label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
        logger.info("Model and label encoder successfully loaded!")
        return model, tokenizer, label_encoder
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise


def init_bot() -> Optional[str]:
    """Initialize bot configuration"""
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("Bot token not found in environment variables")
        raise ValueError("Bot token not found, check your .env file")
    return token


# Load model artifacts
model, tokenizer, label_encoder = load_model_artifacts()
TOKEN = init_bot()
initialize_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


async def analyze_text(update: Update, context: CallbackContext) -> None:
    """Analyze the emotional content of user messages with probabilities"""
    try:
        text = update.message.text.strip()  # type: ignore
        processed_text = preprocess_text(text, lemmatizer, stop_words)
        inputs = tokenizer(
            processed_text, return_tensors="pt", truncation=True, max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            predicted_label = np.argmax(probs)

        emotion = label_encoder.inverse_transform([predicted_label])[0]
        probability = probs[predicted_label] * 100

        emotion_emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😮",
        }
        emoji = emotion_emojis.get(emotion.lower(), "🤔")

        await update.message.reply_text(  # type: ignore
            f"I detect: *{emotion}* {emoji} ({probability:.2f}% confidence)",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        await update.message.reply_text(  # type: ignore
            "Sorry, I encountered an error while analyzing your text. Please try again."
        )


def main() -> None:
    """Initialize and start the bot"""
    try:
        application = ApplicationBuilder().token(TOKEN).build()  # type: ignore
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text)
        )
        logger.info("Bot started successfully!")
        application.run_polling()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise


if __name__ == "__main__":
    main()
