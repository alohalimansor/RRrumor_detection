import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import transformers
import os
import logging
import platform
import multiprocessing

# Add multiprocessing support for Windows
if __name__ == '__main__' and platform.system() == 'Windows':
    # Add freeze_support for Windows
    multiprocessing.freeze_support()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print versions for debugging
logger.info(f"Transformers version: {transformers.__version__}")
logger.info(f"PyTorch version: {torch.__version__}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load the dataset
logger.info("Loading dataset...")
df = pd.read_csv("combined_twitter_rumor_dataset.csv")
logger.info(f"Dataset loaded with {len(df)} rows")

# Preprocessing: Drop missing values, reset index
logger.info("Preprocessing dataset...")
initial_rows = len(df)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")

# Debug: Inspect the 'label' column
logger.info(f"Data type of 'label' column: {df['label'].dtype}")
logger.info(f"Unique values in 'label' column: {df['label'].unique()}")
logger.info(f"Label distribution: \n{df['label'].value_counts()}")

# Handle labels with improved robustness
if df["label"].dtype in [int, float]:
    logger.info("Labels are numeric, ensuring they are integers 0 and 1.")
    df["label"] = df["label"].astype(int)
    logger.info(f"Numeric label counts: {df['label'].value_counts()}")

    # Verify labels are only 0 and 1
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Numeric labels must be 0 or 1. Found: {df['label'].unique()}")
else:
    logger.info("Labels are strings, converting to numeric.")
    # Log original label distribution
    logger.info(f"Original label counts: {df['label'].value_counts()}")

    # Normalize all labels to lowercase and strip whitespace
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    # Expanded mapping to handle variations
    label_mapping = {
        "rumor": 1, "rumors": 1, "true": 1, "1": 1, "yes": 1,
        "non-rumor": 0, "non rumor": 0, "nonrumor": 0, "false": 0, "0": 0, "no": 0
    }

    # Apply mapping
    df["label"] = df["label"].map(label_mapping)

    # Check for unmapped values
    unmapped = df[df["label"].isnull()]
    if not unmapped.empty:
        logger.warning(f"Found {len(unmapped)} rows with unmapped labels:")
        logger.warning(f"Unmapped values: {unmapped['label'].value_counts()}")
        logger.warning("Consider adding these values to your label_mapping dictionary")

        # Drop rows with unmapped labels
        df = df.dropna(subset=["label"])
        logger.info(f"Dropped {len(unmapped)} rows with unmapped labels")

    # Convert to integer type
    df["label"] = df["label"].astype(int)
    logger.info(f"Final label counts after mapping: {df['label'].value_counts()}")

# Check text column
logger.info(f"Text column sample (first 100 chars): {df['text'].iloc[0][:100]}")
logger.info(
    f"Text length stats: min={df['text'].str.len().min()}, max={df['text'].str.len().max()}, avg={df['text'].str.len().mean():.1f}")

# Train-Test Split (80-10-10 split)
logger.info("Splitting dataset into train, validation, and test sets...")
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

logger.info(f"Train set: {len(train_texts)} examples")
logger.info(f"Validation set: {len(val_texts)} examples")
logger.info(f"Test set: {len(test_texts)} examples")

# Calculate class weights for imbalanced dataset
train_class_counts = np.bincount(train_labels)
logger.info(f"Training set class distribution: {train_class_counts}")

if train_class_counts[0] != train_class_counts[1]:
    logger.info("Dataset is imbalanced, calculating class weights...")
    class_weights = len(train_labels) / (2 * train_class_counts)
    logger.info(f"Weight for class 0: {class_weights[0]:.4f}, Weight for class 1: {class_weights[1]:.4f}")
else:
    class_weights = None
    logger.info("Dataset is balanced, no class weights needed.")

# Convert data into Hugging Face Dataset format
logger.info("Creating Hugging Face datasets...")


def create_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": texts.tolist(), "label": labels.tolist()})


train_dataset = create_hf_dataset(train_texts, train_labels)
val_dataset = create_hf_dataset(val_texts, val_labels)
test_dataset = create_hf_dataset(test_texts, test_labels)

# Load XLNet tokenizer and model
logger.info("Loading XLNet tokenizer and model...")
xlnet_model_name = "xlnet-base-cased"
tokenizer_xlnet = AutoTokenizer.from_pretrained(xlnet_model_name)

# Define custom model with weighted loss if needed
if class_weights is not None:
    class XLNetForClassificationWithWeightedLoss(AutoModelForSequenceClassification):
        def __init__(self, config):
            super().__init__(config)
            self.class_weights = torch.tensor([class_weights[0], class_weights[1]])

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

            if labels is not None:
                # Apply class weights to loss
                weights = self.class_weights.to(labels.device)
                loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
                logits = outputs.logits
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
                outputs.loss = loss

            return outputs


    # Create weighted model
    model_xlnet = XLNetForClassificationWithWeightedLoss.from_pretrained(
        xlnet_model_name,
        num_labels=2
    )
    logger.info("Using custom XLNet model with weighted loss")
else:
    # Standard model
    model_xlnet = AutoModelForSequenceClassification.from_pretrained(
        xlnet_model_name,
        num_labels=2
    )
    logger.info("Using standard XLNet model")

# Print model summary
total_params = sum(p.numel() for p in model_xlnet.parameters())
trainable_params = sum(p.numel() for p in model_xlnet.parameters() if p.requires_grad)
logger.info(f"Model loaded with {total_params:,} total parameters, {trainable_params:,} trainable")

# Tokenize the dataset with improved memory management
logger.info("Tokenizing datasets...")


def tokenize_function(examples):
    # Determine max sequence length based on dataset statistics
    # For memory efficiency, we use a smaller max_length
    max_length = min(128, max(64, int(np.percentile([len(text.split()) for text in examples["text"]], 95))))
    logger.info(f"Using max_length of {max_length} tokens for tokenization")

    return tokenizer_xlnet(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


# Process in smaller batches
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32  # Process in smaller chunks
)
val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32
)
test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32
)

# Check a sample of tokenized data
logger.info("Checking first tokenized example:")
example = train_dataset[0]
logger.info(f"Label: {example['label']}")
logger.info(f"Input IDs shape: {len(example['input_ids'])}")

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])


# Define improved metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    # Add confusion matrix for better insight
    conf_mat = confusion_matrix(labels, predictions)
    if conf_mat.size == 4:  # Make sure it's a 2x2 matrix
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
    else:
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


# Determine optimal batch size based on GPU memory
if torch.cuda.is_available():
    # Start with a small batch size and increase to find optimal value
    # This is a simple heuristic - adjust as needed
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    if gpu_mem > 16:
        batch_size = 8
    elif gpu_mem > 8:
        batch_size = 4
    else:
        batch_size = 2
    logger.info(f"GPU memory: {gpu_mem:.2f} GB, using batch size {batch_size}")
else:
    batch_size = 4
    logger.info(f"No GPU detected, using batch size {batch_size}")

# Define Training Arguments with improved configuration
logger.info("Setting up training arguments...")
output_dir = "./xlnet_results"
os.makedirs(output_dir, exist_ok=True)

training_args_xlnet = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs_xlnet",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Use F1 instead of accuracy for imbalanced datasets
    greater_is_better=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
    dataloader_num_workers=0,  # Disable parallel data loading on Windows
    warmup_steps=500,
    seed=42,
    # Add early stopping callback
    push_to_hub=False,
)

# Trainer for XLNet
logger.info("Initializing Trainer...")
trainer_xlnet = Trainer(
    model=model_xlnet,
    args=training_args_xlnet,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train and Evaluate
logger.info("Starting training...")
try:
    trainer_xlnet.train()
    logger.info("Training completed successfully")

    # Save model
    trainer_xlnet.save_model(os.path.join(output_dir, "best_model"))
    tokenizer_xlnet.save_pretrained(os.path.join(output_dir, "best_model"))
    logger.info("Model saved successfully")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    xlnet_results = trainer_xlnet.evaluate(test_dataset)
    logger.info(f"XLNet Test Results: {xlnet_results}")

    # Print comparison
    logger.info("Final Model Comparison on Twitter Rumor Dataset")
    logger.info("RoBERTa Accuracy: 95.84% (Baseline)")
    logger.info(f"XLNet Accuracy: {xlnet_results['eval_accuracy'] * 100:.2f}%")
    logger.info(f"XLNet F1 Score: {xlnet_results['eval_f1'] * 100:.2f}%")

    # Optional: Detailed analysis on test set
    logger.info("Generating predictions for test set...")
    test_predictions = trainer_xlnet.predict(test_dataset)
    preds = np.argmax(test_predictions.predictions, axis=1)

    # Calculate metrics manually for verification
    test_acc = accuracy_score(test_labels, preds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_labels, preds, average='binary')

    logger.info(
        f"Verified Test Metrics - Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")

except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    import traceback

    logger.error(traceback.format_exc())

    """
    Run this script to train the XLNet model for rumor detection.
    This wrapper ensures proper multiprocessing setup on Windows.
    """

    import multiprocessing
    import platform

    # Main entry point for the program
    if __name__ == '__main__':
        # Add freeze_support for Windows
        if platform.system() == 'Windows':
            multiprocessing.freeze_support()

        # Import and run your main script
        import rumor_detection  # Import your main script (save the main code as rumor_detection.py)