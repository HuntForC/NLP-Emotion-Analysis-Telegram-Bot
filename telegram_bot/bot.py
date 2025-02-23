import os
import sys
import joblib
import logging
import torch
import torch.nn.functional as F
from typing import Optional, cast
from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import (
    Update,
    constants,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    CallbackQuery,
    Message,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
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


async def handle_feedback(update: Update, context: CallbackContext) -> None:
    """Handle user feedback on emotion prediction"""
    if not update.callback_query:
        return

    query: CallbackQuery = update.callback_query
    if not query.message or not query.from_user:
        return

    await query.answer()  # Acknowledge the button click

    # Extract the feedback type from callback data
    feedback = query.data or "unknown"
    message = cast(Message, query.message)  # Cast to correct type
    original_message = message.text or ""
    user_name = query.from_user.first_name or "User"

    if feedback == "correct":
        response = f"Thanks {user_name}! 😊 Glad I got it right!"
    else:
        response = f"Thanks {user_name}! 😔 I'll try to do better next time."

    # Edit the original message to remove the buttons
    await query.edit_message_text(
        text=f"{original_message}\n\n{response}",
        parse_mode=constants.ParseMode.MARKDOWN,
        reply_markup=None,
    )


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

        # Create inline keyboard with thumbs up/down buttons
        keyboard = [
            [
                InlineKeyboardButton("👍 Correct", callback_data="correct"),
                InlineKeyboardButton("👎 Incorrect", callback_data="incorrect"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(  # type: ignore
            response, parse_mode=constants.ParseMode.MARKDOWN, reply_markup=reply_markup
        )
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        await update.message.reply_text(  # type: ignore
            "Sorry, I encountered an error while analyzing your text. Please try again."
        )


async def start_command(update: Update, context: CallbackContext) -> None:
    """Handle the /start command"""
    if not update.message:
        return

    welcome_message = (
        "👋 *Welcome to the Emotion Analysis Bot!*\n\n"
        "I can analyze the emotions in your messages using advanced AI. "
        "Simply send me any text message and I'll tell you what emotions I detect.\n\n"
        "Available commands:\n"
        "📝 Just type any message - I'll analyze its emotional content\n"
        "ℹ️ /about - Learn more about how I work\n"
        "❓ /help - Show this help message\n\n"
        "Let's get started! Send me a message to analyze. 😊"
    )
    await update.message.reply_text(
        welcome_message, parse_mode=constants.ParseMode.MARKDOWN
    )


async def about_command(update: Update, context: CallbackContext) -> None:
    """Handle the /about command"""
    if not update.message:
        return

    about_message = (
        "🤖 *About Emotion Analysis Bot*\n\n"
        "I'm powered by advanced Natural Language Processing and Machine Learning technology. "
        "I use a fine-tuned BERT model to analyze emotions in text messages.\n\n"
        "I can detect the following emotions:\n"
        "• Joy 😊\n"
        "• Sadness 😢\n"
        "• Anger 😠\n"
        "• Fear 😨\n"
        "• Love ❤️\n"
        "• Surprise 😮\n\n"
        "Your feedback helps me improve! After each analysis, you can let me know if I got it right."
    )
    await update.message.reply_text(
        about_message, parse_mode=constants.ParseMode.MARKDOWN
    )


async def help_command(update: Update, context: CallbackContext) -> None:
    """Handle the /help command"""
    await start_command(update, context)


def main() -> None:
    """Initialize and start the bot"""
    try:
        application = ApplicationBuilder().token(TOKEN).build()  # type: ignore

        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("about", about_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text)
        )
        application.add_handler(CallbackQueryHandler(handle_feedback))

        logger.info("Bot started successfully!")
        application.run_polling()
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise


if __name__ == "__main__":
    main()
