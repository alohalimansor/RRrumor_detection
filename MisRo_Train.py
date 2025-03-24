import torch
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import os
import logging
import platform
import multiprocessing
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

# Main execution guard for Windows compatibility
if __name__ == '__main__' and platform.system() == 'Windows':
    multiprocessing.freeze_support()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"misrobaerta_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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

# Split dataset with stratification
logger.info("Splitting dataset...")
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

logger.info(f"Train set: {len(train_texts)} examples")
logger.info(f"Validation set: {len(val_texts)} examples")
logger.info(f"Test set: {len(test_texts)} examples")

# Check class distribution in splits
logger.info(f"Train label distribution: {pd.Series(train_labels).value_counts()}")
logger.info(f"Validation label distribution: {pd.Series(val_labels).value_counts()}")
logger.info(f"Test label distribution: {pd.Series(test_labels).value_counts()}")

# Compute class weights if imbalanced
class_counts = pd.Series(train_labels).value_counts()
if max(class_counts) / min(class_counts) > 1.5:
    logger.info("Dataset is imbalanced, computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    weight_dict = {i: float(weight) for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {weight_dict}")
else:
    weight_dict = None
    logger.info("Dataset is relatively balanced, no class weights applied.")

# Convert to Hugging Face Dataset format
logger.info("Creating Hugging Face datasets...")
def create_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": texts.tolist(), "label": labels.tolist()})

train_dataset = create_hf_dataset(train_texts, train_labels)
val_dataset = create_hf_dataset(val_texts, val_labels)

# Load tokenizer
logger.info("Loading tokenizer...")
model_name = "digitalepidemiologylab/MisRoBÆRTa"  # Placeholder; not on Hugging Face Hub
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Error loading tokenizer: {str(e)}")
    logger.info("Falling back to 'roberta-base' tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=32)
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=32)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

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

# Create output directory
output_dir = "./misrobaerta_results"
os.makedirs(output_dir, exist_ok=True)

# Load model
logger.info("Loading model for training...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.info("Falling back to 'roberta-base' model")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model loaded with {total_params:,} parameters")

# Define metrics for validation during training
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Custom Trainer for class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(list(self.class_weights.values()), dtype=torch.float).to(device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Define training arguments
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs_misrobaerta",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    seed=42,
)

# Initialize trainer
logger.info("Initializing trainer...")
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=weight_dict if weight_dict else None,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Train the model
logger.info("Starting training...")
start_time = time.time()
try:
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    logger.info("Model and tokenizer saved successfully to {output_dir}/best_model")
except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise