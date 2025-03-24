import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import transformers
from datasets import Dataset
import os
import logging
import platform
import multiprocessing

if __name__ == '__main__' and platform.system() == 'Windows':
    multiprocessing.freeze_support()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Transformers version: {transformers.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv("combined_twitter_rumor_dataset.csv")
logger.info(f"Dataset loaded with {len(df)} rows")
logger.info(f"Column names: {df.columns.tolist()}")
logger.info(f"Initial missing values:\n{df.isnull().sum()}")

# Preprocessing
logger.info("Preprocessing dataset...")
df.dropna(subset=["text", "label"], inplace=True)
logger.info(f"After dropna: {len(df)} rows")
df.reset_index(drop=True, inplace=True)

logger.info(f"Label column dtype: {df['label'].dtype}")
logger.info(f"Unique label values: {df['label'].unique()}")

if df["label"].dtype not in [np.int64, np.int32, np.float64, np.float32]:
    logger.info("Labels are strings, converting to numeric...")
    label_mapping = {"rumor": 1, "non-rumor": 0}
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["label"] = df["label"].map(label_mapping)
    if df["label"].isnull().any():
        df.dropna(subset=["label"], inplace=True)
        logger.info(f"Dropped rows with unmapped labels, remaining: {len(df)}")
else:
    logger.info("Labels are already numeric, ensuring integer format.")
    df["label"] = df["label"].astype(int)

logger.info(f"Label distribution after processing: {df['label'].value_counts()}")

if len(df) == 0:
    logger.error("DataFrame is empty after preprocessing.")
    raise ValueError("Empty DataFrame after preprocessing.")

# Split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

train_df["text_input"] = "classify rumor: " + train_df["text"]
train_df["text_output"] = train_df["label"].map({0: "false", 1: "true"})
val_df["text_input"] = "classify rumor: " + val_df["text"]
val_df["text_output"] = val_df["label"].map({0: "false", 1: "true"})
test_df["text_input"] = "classify rumor: " + test_df["text"]
test_df["text_output"] = test_df["label"].map({0: "false", 1: "true"})

logger.info(f"Train set: {len(train_df)} examples")
logger.info(f"Validation set: {len(val_df)} examples")
logger.info(f"Test set: {len(test_df)} examples")

# Save test set for evaluation
test_df.to_csv("test_dataset.csv", index=False)
logger.info("Test set saved to test_dataset.csv")

# Convert to Hugging Face Dataset
def create_hf_dataset(df):
    return Dataset.from_dict({
        "input_text": df["text_input"].tolist(),
        "target_text": df["text_output"].tolist()
    })

train_dataset = create_hf_dataset(train_df)
val_dataset = create_hf_dataset(val_df)

# Load T5 tokenizer
logger.info("Loading T5 tokenizer...")
t5_model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# Tokenize function
def tokenize_t5_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["target_text"],
            max_length=8,
            padding="max_length",
            truncation=True
        )
    inputs["labels"] = targets["input_ids"]
    inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in inputs["labels"]
    ]
    return inputs

# Tokenize datasets
logger.info("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_t5_function, batched=True, batch_size=16)
val_dataset = val_dataset.map(tokenize_t5_function, batched=True, batch_size=16)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load T5 model
logger.info("Loading T5 model...")
model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
model.to(device)

# Set batch size
batch_size = 4 if torch.cuda.is_available() else 2
logger.info(f"Using batch size: {batch_size}")

# Create output directory
output_dir = "./t5_results"
os.makedirs(output_dir, exist_ok=True)

# Define training arguments (no evaluation during training)
training_args = TrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=5e-5,
    logging_dir="./logs_t5",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    seed=42,
)

# Custom Trainer
class T5GenerationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t5_tokenizer = tokenizer

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=8,
                num_beams=2
            )
            if prediction_loss_only:
                return (None, None, None)
            labels = inputs.get("labels", None)
            return (None, generated_ids, labels)

# Initialize trainer
logger.info("Initializing T5 trainer...")
trainer = T5GenerationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset  # Still included but not used during training
)

# Train
logger.info("Starting training...")
trainer.train()
logger.info("Training completed successfully")
trainer.save_model(os.path.join(output_dir, "best_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
logger.info("Model saved successfully")