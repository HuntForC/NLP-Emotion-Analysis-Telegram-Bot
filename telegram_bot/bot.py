import os
import joblib
import logging
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from telegram import Update, constants
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Constants and configurations
class Config:
    """Bot configuration settings"""

    MODEL_DIR = Path("model_training")
    MODEL_PATH = MODEL_DIR / "model.pkl"
    LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
    HELP_MESSAGE = """
I can analyze the emotional content of your messages! 🎭

Commands:
/start - Start the bot
/help - Show this help message
/about - Learn more about how I work

Just send me any text and I'll determine its emotional tone.
"""
    ABOUT_MESSAGE = """
I'm an NLP-powered Emotion Analysis bot! 🤖

I use machine learning to analyze the emotional content of text messages.
I can detect various emotions in your text and help you understand the emotional tone of your messages.

Send me any text to try it out!
"""


def load_model_artifacts() -> tuple[Pipeline, LabelEncoder]:
    """Load the trained model and label encoder"""
    try:
        if not Config.MODEL_PATH.exists() or not Config.LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError("Model files not found")

        model: Pipeline = joblib.load(Config.MODEL_PATH)
        label_encoder: LabelEncoder = joblib.load(Config.LABEL_ENCODER_PATH)
        logger.info("Model and label encoder successfully loaded!")
        return model, label_encoder
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
model, label_encoder = load_model_artifacts()
TOKEN = init_bot()


async def start_command(update: Update, context: CallbackContext) -> None:
    """Handle the /start command"""
    user = update.effective_user
    await update.message.reply_text(  # type: ignore
        f"Hello {user.first_name}! 👋\n\n"  # type: ignore
        "I can analyze the emotional content of your messages.\n"
        "Send me any text and I'll determine its mood!\n\n"
        "Use /help to see all available commands.",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def help_command(update: Update, context: CallbackContext) -> None:
    """Handle the /help command"""
    await update.message.reply_text(  # type: ignore
        Config.HELP_MESSAGE, parse_mode=constants.ParseMode.MARKDOWN
    )


async def about_command(update: Update, context: CallbackContext) -> None:
    """Handle the /about command"""
    await update.message.reply_text(  # type: ignore
        Config.ABOUT_MESSAGE, parse_mode=constants.ParseMode.MARKDOWN
    )


async def analyze_text(update: Update, context: CallbackContext) -> None:
    """Analyze the emotional content of user messages"""
    try:
        text = update.message.text  # type: ignore

        # Predict emotion
        predicted_label = model.predict([text])[0]
        emotion = label_encoder.inverse_transform([predicted_label])[0]

        # Map emotions to emojis
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
            f"I detect: *{emotion}* {emoji}", parse_mode=constants.ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        await update.message.reply_text(  # type: ignore
            "Sorry, I encountered an error while analyzing your text. Please try again."
        )


async def error_handler(update: Update, context: CallbackContext) -> None:
    """Handle errors in the bot"""
    logger.error(f"Update {update} caused error {context.error}")
    if update.message:
        await update.message.reply_text(  # type: ignore
            "Sorry, something went wrong. Please try again later."
        )


def main() -> None:
    """Initialize and start the bot"""
    try:
        # Create application
        application = ApplicationBuilder().token(TOKEN).build()  # type: ignore

        # Add command handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("about", about_command))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text)
        )

        # Add error handler
        application.add_error_handler(error_handler)  # type: ignore

        # Start the bot
        logger.info("Bot started successfully!")
        application.run_polling()

    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise


if __name__ == "__main__":
    main()
