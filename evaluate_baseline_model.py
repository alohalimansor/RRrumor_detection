import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the test dataset
df = pd.read_csv("combined_twitter_rumor_dataset.csv")

# Clean the dataset by removing rows with missing text and extract relevant columns
df_clean = df.dropna(subset=['text'])[['text', 'label']]
df_clean['label'] = df_clean['label'].astype(int)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and encode the test data
test_encodings = tokenizer(df_clean['text'].tolist(), truncation=True, padding=True, max_length=512)


# Create a PyTorch Dataset for efficient batching
class RumorDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# Create the test dataset and data loader
test_dataset = RumorDataset(test_encodings, df_clean['label'].tolist())
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the pre-trained BERT model for sequence classification with 2 classes
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# Define evaluation function to generate predictions
def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data and return predictions and true labels

    Args:
        model: Hugging Face transformer model
        test_loader: DataLoader containing test data

    Returns:
        predictions: List of model predictions
        true_labels: List of ground truth labels
    """
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)

    return predictions, true_labels


# Run evaluation on the test data
predictions, true_labels = evaluate_model(model, test_loader)

# Compute accuracy and detailed classification metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=["Non-Rumor", "Rumor"])

print(f"Test Accuracy: {accuracy:.4f}")
print(report)

# Generate and visualize confusion matrix
cm = confusion_matrix(true_labels, predictions)
labels = ["Non-Rumor", "Rumor"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Analyze class distribution in model predictions
unique, counts = np.unique(predictions, return_counts=True)
predicted_distribution = dict(zip(unique, counts))

print(f"Predicted Class Distribution: {predicted_distribution}")