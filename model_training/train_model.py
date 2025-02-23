import joblib
import torch
from pathlib import Path
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import prepare_datasets

# File paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_training/emotion_bert"
LABEL_ENCODER_PATH = BASE_DIR / "model_training/label_encoder.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data/train.txt"
VAL_DATA_PATH = BASE_DIR / "data/val.txt"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EmotionTrainingCallback(TrainerCallback):
    """Custom callback to display training progress and metrics"""

    def __init__(self, label_encoder):
        self.label_encoder = label_encoder

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        print(f"\nStarting epoch {int(epoch) + 1}/{args.num_train_epochs}")

    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        """Display metrics at the end of each epoch"""
        epoch = state.epoch if state.epoch is not None else 0
        print(f"\nEpoch {int(epoch) + 1} completed")
        if metrics:
            # Display training metrics
            train_loss = metrics.get("train_loss", 0)
            print(f"Training loss: {train_loss:.4f}")

            # Display evaluation metrics if available
            eval_loss = metrics.get("eval_loss")
            if eval_loss:
                print(f"Validation loss: {eval_loss:.4f}")


def compute_metrics(label_encoder):
    """Create compute_metrics function with access to label_encoder"""

    def compute_metrics_with_labels(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)

        # Get classification report
        report = classification_report(
            labels,
            predictions,
            target_names=label_encoder.classes_,
            digits=4,
            zero_division=0,
        )

        print("\nClassification Report:")
        print(report)

        return {
            "accuracy": accuracy,
        }

    return compute_metrics_with_labels


def train_and_save_model():
    """Train BERT model and save artifacts"""
    # Prepare datasets
    train_dataset, val_dataset, label_encoder = prepare_datasets(
        TRAIN_DATA_PATH, VAL_DATA_PATH
    )

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(label_encoder.classes_)
    )

    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors=None,
        )
        # Preserve the labels
        tokenized["labels"] = examples["label"]
        return tokenized

    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in train_dataset.column_names if col != "label"],
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in val_dataset.column_names if col != "label"],
    )

    # Set format for pytorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Training arguments optimized for maximum speed
    training_args = TrainingArguments(
        output_dir=str(RESULTS_DIR),
        num_train_epochs=2,  # Reduced from 3
        # Larger batch size
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        # Gradient accumulation
        gradient_accumulation_steps=2,
        # Faster learning rate
        learning_rate=1e-4,
        warmup_ratio=0.05,
        # Reduced weight decay
        weight_decay=0.001,
        # Minimal logging
        logging_dir=str(LOGS_DIR),
        logging_steps=50,
        # Evaluation and save strategies aligned to steps
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        # Mixed precision training
        fp16=torch.cuda.is_available(),
        half_precision_backend="auto",
        # Load best model
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # Report to tensorboard
        report_to=["tensorboard"],
        # Other optimizations
        disable_tqdm=False,
        dataloader_num_workers=4,  # Parallel data loading
        group_by_length=True,  # Group similar length sequences
    )

    # Initialize trainer with callback and metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics(label_encoder),
        callbacks=[EmotionTrainingCallback(label_encoder)],
    )

    print("\nStarting model training with optimized settings...")
    print(f"Training on {len(train_dataset)} examples")
    print(f"Validating on {len(val_dataset)} examples")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(
        f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )
    print(
        f"Mixed precision training: {'Enabled' if training_args.fp16 else 'Disabled'}"
    )
    print("=" * 50)

    # Train the model
    trainer.train()

    # Final evaluation
    print("\nFinal Evaluation:")
    trainer.evaluate()

    # Save model and tokenizer
    model.save_pretrained(MODEL_PATH, safe_serialization=False)  # Disable safetensors
    # Also save state dict separately for backup
    torch.save(model.state_dict(), MODEL_PATH / "pytorch_model.bin")
    tokenizer.save_pretrained(MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print("\nModel trained and saved successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    train_and_save_model()
