import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Load dataset
file_path = "cleaned_rumors.csv"  # Ensure this is the correct file name in your PyCharm project
df = pd.read_csv(file_path, encoding="utf-8")

# Verify dataset structure
print("Dataset Sample Before Processing:")
print(df.head())

# Use only the cleaned text column
if "clean_text" in df.columns:
    texts = df["clean_text"].dropna().tolist()
else:
    texts = df.iloc[:, 1].dropna().tolist()  # Assuming second column is cleaned text

# Load RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Define a PyTorch Dataset for MLM
class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Create dataset
mlm_dataset = MLMDataset(texts, tokenizer)

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Load RoBERTa model for Masked Language Modeling
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mlm_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=3,  # You can increase this if needed
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mlm_dataset,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("./pretrained_roberta_mlm")
tokenizer.save_pretrained("./pretrained_roberta_mlm")

print("\nSelf-Supervised Pretraining Completed! Model saved as 'pretrained_roberta_mlm'.")