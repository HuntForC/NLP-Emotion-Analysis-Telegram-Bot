import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load and parse dataset file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, emotion = line.strip().split(';')
            data.append({'text': text, 'emotion': emotion})
    return pd.DataFrame(data)

def preprocess_data(df):
    """Encode emotions to numerical labels"""
    le = LabelEncoder()
    df['emotion_encoded'] = le.fit_transform(df['emotion'])
    return df, le