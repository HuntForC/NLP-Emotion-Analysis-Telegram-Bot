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
    MODEL_STATE_DICT_PATH = (
        MODEL_DIR / "pytorch_model.bin"
    )  # Add path for PyTorch model


def load_model_artifacts():
    """Load the trained model and label encoder"""
    try:
        # First try loading with default method
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                Config.MODEL_PATH,
                local_files_only=True,
                use_safetensors=False,  # Explicitly disable safetensors
            )
        except Exception as e:
            logger.warning(f"Failed to load model with default method: {e}")
            # Fallback to loading base model and state dict separately
            config_path = Config.MODEL_PATH / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")

            # Initialize model from base architecture
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                config=config_path,
                local_files_only=False,  # Allow downloading base model
            )

            # Load state dict if available
            if Config.MODEL_STATE_DICT_PATH.exists():
                state_dict = torch.load(Config.MODEL_STATE_DICT_PATH)
                model.load_state_dict(state_dict)
            else:
                logger.warning("Model state dict not found, using base model")

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
            # Get indices of top 2 predictions
            top_2_indices = np.argsort(probs)[-2:][::-1]

        # Get emotions and probabilities for top 2 predictions
        emotions = label_encoder.inverse_transform(top_2_indices)
        probabilities = probs[top_2_indices] * 100

        emotion_emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😮",
        }

        # Format response with top 2 emotions
        main_emotion = emotions[0]
        main_emoji = emotion_emojis.get(main_emotion.lower(), "🤔")
        second_emotion = emotions[1]
        second_emoji = emotion_emojis.get(second_emotion.lower(), "🤔")

        response = (
            f"I detect: *{main_emotion}* {main_emoji} ({probabilities[0]:.1f}%)\n"
            f"But maybe: *{second_emotion}* {second_emoji} ({probabilities[1]:.1f}%)"
        )

        await update.message.reply_text(  # type: ignore
            response,
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
