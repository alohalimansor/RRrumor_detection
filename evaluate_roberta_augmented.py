import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load labeled dataset for testing
file_path = "augmented_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])

# Load tokenizer for the fine-tuned augmented RoBERTa model
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_roberta_augmented")

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

# Load the fine-tuned augmented RoBERTa model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_roberta_augmented").to(device)

# Run evaluation in inference mode
print("Starting evaluation on the augmented model...")
model.eval()
predictions = []
true_labels = []
pred_probs = []

with torch.no_grad():
    for batch in test_dataset:
        # Process each item individually
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}  # Add batch dimension
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()  # Probability of rumor (class 1)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels)
        pred_probs.extend(probs)

# Convert to numpy arrays for analysis
true_labels = np.array(true_labels)
predictions = np.array(predictions)
pred_probs = np.array(pred_probs)

# Print evaluation metrics
print("Test Accuracy:", accuracy_score(true_labels, predictions))
print(classification_report(true_labels, predictions, target_names=["Non-Rumor", "Rumor"]))

# Create a combined figure with subplots for visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix (Subplot 1)
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Rumor', 'Rumor'],
            yticklabels=['Non-Rumor', 'Rumor'], annot_kws={"size": 14}, vmin=0, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=16)
ax1.set_ylabel('True Label', fontsize=14)
ax1.set_xlabel('Predicted Label', fontsize=14)
ax1.text(0, 0, 'TN', ha='center', va='center', color='white', fontsize=12)
ax1.text(0, 1, 'FN', ha='center', va='center', color='black', fontsize=12)
ax1.text(1, 0, 'FP', ha='center', va='center', color='black', fontsize=12)
ax1.text(1, 1, 'TP', ha='center', va='center', color='white', fontsize=12)

# ROC Curve (Subplot 2)
fpr, tpr, _ = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.0])
ax2.set_xlabel('False Positive Rate', fontsize=14)
ax2.set_ylabel('True Positive Rate', fontsize=14)
ax2.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
ax2.legend(loc="lower right", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('combined_performance.png', dpi=300)
plt.show()

print("Evaluation completed!")