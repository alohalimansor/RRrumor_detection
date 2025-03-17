import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load labeled dataset
file_path = "combined_twitter_rumor_dataset.csv"  # Ensure this file is in your project directory
df = pd.read_csv(file_path, encoding="utf-8")

# Verify dataset structure
print("Dataset Sample Before Processing:")
print(df.head())

# Filter rows where both 'text' and 'label' are non-null
df = df.dropna(subset=["text", "label"])  # Drop rows where either is null
print(f"Rows after dropping nulls: {len(df)}")

# Ensure the dataset has 'text' and 'label' columns
if "text" in df.columns and "label" in df.columns:
    texts = df["text"].tolist()  # No need for dropna() now
    labels = df["label"].astype(int).tolist()  # Convert to int
else:
    raise ValueError("Dataset must have 'text' and 'label' columns.")

# Verify lengths match
assert len(texts) == len(labels), f"Length mismatch: texts={len(texts)}, labels={len(labels)}"

# Split dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Load the tokenizer from the pre-trained model
tokenizer = AutoTokenizer.from_pretrained("./pretrained_roberta_mlm")

# Define PyTorch Dataset
class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Create dataset objects
train_dataset = RumorDataset(train_texts, train_labels, tokenizer)
test_dataset = RumorDataset(test_texts, test_labels, tokenizer)

# Load the pre-trained model and set it up for classification
model = AutoModelForSequenceClassification.from_pretrained("./pretrained_roberta_mlm", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_roberta",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_roberta")
tokenizer.save_pretrained("./fine_tuned_roberta")

print("\nFine-Tuning Completed! Model saved as 'fine_tuned_roberta'.")

# Evaluate the model
def evaluate_model():
    # Set model to evaluation mode and compute predictions
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_dataset:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}  # Add batch dimension
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)
    print("\nTest Accuracy:", accuracy_score(true_labels, predictions))
    print(classification_report(true_labels, predictions, target_names=["Non-Rumor", "Rumor"]))

evaluate_model()