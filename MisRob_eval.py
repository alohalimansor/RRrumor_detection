import torch
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os
import logging
import platform
import multiprocessing
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Main execution guard for Windows compatibility
if __name__ == '__main__' and platform.system() == 'Windows':
    multiprocessing.freeze_support()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"misrobaerta_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log system info
logger.info(f"Python version: {platform.python_version()}")
logger.info(f"PyTorch version: {torch.__version__}")
import transformers
logger.info(f"Transformers version: {transformers.__version__}")

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Function to preprocess tweets (vectorized)
def preprocess_tweet_vectorized(df):
    logger.info("Applying vectorized tweet preprocessing...")
    df['text'] = df['text'].astype(str).str.lower()
    df['text'] = df['text'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)
    df['text'] = df['text'].str.replace(r'#', '', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

# Load and validate dataset
dataset_path = "combined_twitter_rumor_dataset.csv"
logger.info("Validating dataset...")
if os.path.exists(dataset_path):
    logger.info(f"CSV file exists at {dataset_path}")
    file_size = os.path.getsize(dataset_path)
    logger.info(f"File size: {file_size / 1024:.2f} KB ({file_size} bytes)")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        logger.info(f"CSV header: {header}")
        for i in range(5):
            sample = f.readline().strip()
            if sample:
                logger.info(f"Sample line {i + 1}: {sample[:100]}...")
else:
    logger.error(f"CSV file does not exist at {dataset_path}")
    raise FileNotFoundError(f"CSV file not found: {dataset_path}")

logger.info("Loading dataset...")
try:
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded with {len(df)} rows")
    logger.info(f"Initial dataset shape: {df.shape}")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# Preprocessing
logger.info("Preprocessing dataset...")
initial_rows = len(df)
df.dropna(subset=["text", "label"], inplace=True)
logger.info(f"Shape after dropping NA values: {df.shape}")
df.reset_index(drop=True, inplace=True)
logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
df = preprocess_tweet_vectorized(df)

# Validate labels (numeric, 0 and 1 expected)
logger.info(f"Data type of 'label' column: {df['label'].dtype}")
logger.info(f"Unique values in 'label' column: {df['label'].unique()}")
logger.info(f"Label distribution: \n{df['label'].value_counts()}")
df["label"] = df["label"].astype(int)
unique_labels = df["label"].unique()
if not set(unique_labels).issubset({0, 1}):
    logger.error(f"Labels must be 0 or 1, found: {unique_labels}")
    raise ValueError(f"Invalid labels detected: {unique_labels}")
if len(unique_labels) < 2:
    logger.warning(f"Only one class found: {unique_labels}")

# Split dataset with stratification (recreate splits for consistency)
logger.info("Splitting dataset...")
_, temp_texts, _, temp_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

logger.info(f"Validation set: {len(val_texts)} examples")
logger.info(f"Test set: {len(test_texts)} examples")

# Convert to Hugging Face Dataset format
logger.info("Creating Hugging Face datasets...")
def create_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": texts.tolist(), "label": labels.tolist()})

val_dataset = create_hf_dataset(val_texts, val_labels)
test_dataset = create_hf_dataset(test_texts, test_labels)

# Load saved model and tokenizer
output_dir = "./misrobaerta_results"
model_path = os.path.join(output_dir, "best_model")
if not os.path.exists(model_path):
    logger.error(f"Trained model not found at {model_path}. Please run train.py first.")
    raise FileNotFoundError(f"Model directory {model_path} not found")

logger.info("Loading saved model and tokenizer...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading saved model or tokenizer: {str(e)}")
    raise

model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model loaded with {total_params:,} parameters")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

logger.info("Tokenizing datasets...")
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=32)
test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=32)

# Set format for PyTorch
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define metrics for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    conf_mat = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = conf_mat.ravel()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

# Set batch size based on GPU memory
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gpu_mem > 16:
        batch_size = 16
    elif gpu_mem > 8:
        batch_size = 8
    elif gpu_mem > 4:
        batch_size = 4
    else:
        batch_size = 2
    logger.info(f"GPU memory: {gpu_mem:.2f} GB, using batch size {batch_size}")
else:
    batch_size = 4
    logger.info(f"No GPU detected, using batch size {batch_size}")

# Initialize trainer for evaluation
logger.info("Initializing trainer for evaluation...")
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs_misrobaerta",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    ),
    compute_metrics=compute_metrics,
)

# Evaluation functions
def plot_training_history(logs, output_dir):
    train_logs = logs[logs['loss'].notna()] if 'loss' in logs else pd.DataFrame()
    eval_logs = logs[logs['eval_loss'].notna()] if 'eval_loss' in logs else pd.DataFrame()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    if not train_logs.empty:
        axs[0, 0].plot(train_logs['epoch'], train_logs['loss'])
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].grid(True)
    if not eval_logs.empty:
        axs[0, 1].plot(eval_logs['epoch'], eval_logs['eval_loss'], label='Loss')
        axs[0, 1].set_title('Validation Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
        axs[1, 0].plot(eval_logs['epoch'], eval_logs['eval_f1'], color='green')
        axs[1, 0].set_title('Validation F1 Score')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('F1 Score')
        axs[1, 0].grid(True)
        axs[1, 1].plot(eval_logs['epoch'], eval_logs['eval_accuracy'], color='purple')
        axs[1, 1].set_title('Validation Accuracy')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_eval.png'))
    plt.close()

def plot_confusion_matrix(conf_matrix, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Rumor', 'Rumor'],
        yticklabels=['Non-Rumor', 'Rumor']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Evaluate
logger.info("Starting evaluation...")
start_time = time.time()
try:
    # Load training history if available (optional)
    if os.path.exists(os.path.join(output_dir, "trainer_state.json")):
        import json
        with open(os.path.join(output_dir, "trainer_state.json"), 'r') as f:
            trainer_state = json.load(f)
        logs = pd.DataFrame(trainer_state.get("log_history", []))
        if not logs.empty:
            plot_training_history(logs, output_dir)
            logger.info("Training history plotted from saved logs")

    logger.info("Evaluating on validation set...")
    val_results = trainer.evaluate(val_dataset)
    logger.info(f"Validation Results: {val_results}")

    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test Results: {test_results}")

    test_predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(test_predictions.predictions, axis=1)
    y_true = test_predictions.label_ids

    conf_mat = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{conf_mat}")
    try:
        plot_confusion_matrix(conf_mat, output_dir)
        logger.info("Confusion matrix plotted successfully")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")

    class_report = classification_report(y_true, y_pred, target_names=['Non-Rumor', 'Rumor'])
    logger.info(f"Classification Report:\n{class_report}")

    eval_time = time.time() - start_time
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Evaluation Time: {eval_time:.2f} seconds ({eval_time / 60:.2f} minutes)\n\n")
        f.write(f"Test Results:\n{test_results}\n\n")
        f.write(f"Confusion Matrix:\n{conf_mat}\n\n")
        f.write(f"Classification Report:\n{class_report}\n")

    accuracy = test_results.get("eval_accuracy", 0) * 100
    precision = test_results.get("eval_precision", 0) * 100
    recall = test_results.get("eval_recall", 0) * 100
    f1 = test_results.get("eval_f1", 0) * 100

    logger.info("=" * 50)
    logger.info("FINAL EVALUATION RESULTS FOR RUMOR DETECTION TASK")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.2f}%")
    logger.info(f"Recall:    {recall:.2f}%")
    logger.info(f"F1 Score:  {f1:.2f}%")
    logger.info("=" * 50)

except Exception as e:
    logger.error(f"Error during evaluation: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise