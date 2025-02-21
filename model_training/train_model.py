import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from data_preprocessing import load_data, preprocess_data

def train_model():
    # Load and preprocess data
    train_df = load_data('data/train.txt')
    train_df, label_encoder = preprocess_data(train_df)
    
    # Create model pipeline with best parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
        ('clf', LogisticRegression(C=10.0, max_iter=1000, solver='lbfgs'))
    ])
    
    # Train model
    pipeline.fit(train_df['text'], train_df['emotion_encoded'])
    
    # Evaluate on validation set
    # test_df = load_data('data/val.txt')
    # test_df, _ = preprocess_data(test_df)
    # y_pred = pipeline.predict(test_df['text'])
    # accuracy = accuracy_score(test_df['emotion_encoded'], y_pred)
    # print(f"\nValidation set accuracy: {accuracy:.4f}")

    # Save artifacts
    joblib.dump(pipeline, 'model_training/model.pkl')
    joblib.dump(label_encoder, 'model_training/label_encoder.pkl')
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()

