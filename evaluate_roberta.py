import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report

# Load labeled dataset for testing
file_path = "combined_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])

# Load tokenizer for the fine-tuned balanced RoBERTa model
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_roberta_balanced")

# Define PyTorch Dataset class for efficient data handling
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

# Prepare test dataset with texts and labels
test_texts = df["text"].tolist()
test_labels = df["label"].astype(int).tolist()
test_dataset = RumorDataset(test_texts, test_labels, tokenizer)

# Load the fine-tuned RoBERTa model with class balancing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_roberta_balanced").to(device)

# Run evaluation in inference mode
print("Starting evaluation...")
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_dataset:
        # Process each item individually (could be optimized with DataLoader for batch processing)
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}  # Add batch dimension
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels)

# Print evaluation metrics
print("\nTest Accuracy:", accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions, target_names=["Non-Rumor", "Rumor"]))

print("\nEvaluation completed!")