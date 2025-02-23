# NLP-Emotion-Analysis-Telegram-Bot

## Description

A sophisticated Telegram bot powered by BERT (Bidirectional Encoder Representations from Transformers) that performs real-time emotion analysis on text messages. The bot uses advanced Natural Language Processing (NLP) techniques to detect and classify emotions in messages, providing detailed probability scores and allowing user feedback for continuous improvement.

## Features

- **Advanced Emotion Detection**:

  - Analyzes 6 basic emotions: Joy 😊, Sadness 😢, Anger 😠, Fear 😨, Love ❤️, Surprise 😮
  - Provides confidence scores for predictions
  - Shows top 2 most likely emotions for each message
  - Uses emojis for better visualization

- **Interactive Interface**:

  - User feedback system with 👍/👎 reactions
  - Personalized responses using user's name
  - Markdown-formatted messages for clear presentation

- **Bot Commands**:

  - `/start` - Welcome message and usage instructions
  - `/about` - Information about the bot's capabilities
  - `/help` - Display available commands
  - Direct message analysis - Any text message is automatically analyzed

- **Technical Features**:
  - BERT-based model for accurate emotion classification
  - NLTK preprocessing with lemmatization and stopword removal
  - Comprehensive error handling and logging
  - Environment-based configuration

## Prerequisites

- Python 3.8 or higher
- Telegram Bot Token (obtain from [@BotFather](https://t.me/botfather))
- Internet connection

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/NLP-Emotion-Analysis-Telegram-Bot.git
cd NLP-Emotion-Analysis-Telegram-Bot
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Telegram Bot Token:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

## Usage

1. Start the bot:

```bash
python bot.py
```

2. Open Telegram and search for your bot using its username

3. Interact with the bot:
   - Send any text message to get emotion analysis
   - Use commands like `/start`, `/about`, or `/help`
   - Provide feedback on predictions using 👍/👎 buttons

## Examples

1. Direct Message Analysis:

   ```
   User: "I just got promoted at work!"
   Bot: I detect: *Joy* 😊 (85.5%)
        But maybe: *Love* ❤️ (12.3%)
   ```

2. Using Commands:
   - `/start` - Get welcome message and instructions
   - `/about` - Learn about the bot's capabilities
   - `/help` - View available commands

## Technical Architecture

- **Model**: Fine-tuned BERT (DistilBERT base)
- **NLP Processing**: NLTK for text preprocessing
- **Framework**: python-telegram-bot for Telegram integration
- **Storage**: Local model artifacts and label encoders
