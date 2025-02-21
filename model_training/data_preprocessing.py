import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def initialize_nltk():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

def preprocess_text(text, lemmatizer, stop_words):
    """Tokenize, remove stopwords and lemmatize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    return ' '.join(tokens)

def load_data(file_path):
    """Load and parse dataset file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, emotion = line.strip().split(';')
            data.append({'text': text, 'emotion': emotion})
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess text data and encode emotions"""
    # Initialize NLTK resources
    initialize_nltk()
    
    # Initialize preprocessing tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Preprocess text
    df['text'] = df['text'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
    
    # Encode emotions
    le = LabelEncoder()
    df['emotion_encoded'] = le.fit_transform(df['emotion'])
    
    return df, le