import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import os

# Load dataset
df = pd.read_csv("combined_twitter_rumor_dataset.csv")

# Drop missing values and select relevant columns
df_clean = df.dropna(subset=['text'])[['text', 'label']]
df_clean['label'] = df_clean['label'].astype(int)

# Split into training (80%) and testing (20%) sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_clean['text'].tolist(), df_clean['label'].tolist(), test_size=0.2, random_state=42, stratify=df_clean['label']
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load tokenizers for each model
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")


# Tokenize datasets
train_encodings_bert = bert_tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings_bert = bert_tokenizer(test_texts, truncation=True, padding=True, max_length=512)

train_encodings_roberta = roberta_tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings_roberta = roberta_tokenizer(test_texts, truncation=True, padding=True, max_length=512)

train_encodings_deberta = deberta_tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings_deberta = deberta_tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert data into PyTorch Dataset
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

# Create dataset objects
train_dataset_bert = RumorDataset(train_encodings_bert, train_labels)
test_dataset_bert = RumorDataset(test_encodings_bert, test_labels)

train_dataset_roberta = RumorDataset(train_encodings_roberta, train_labels)
test_dataset_roberta = RumorDataset(test_encodings_roberta, test_labels)

train_dataset_deberta = RumorDataset(train_encodings_deberta, train_labels)
test_dataset_deberta = RumorDataset(test_encodings_deberta, test_labels)

# Define DataLoaders
train_loader_bert = DataLoader(train_dataset_bert, batch_size=8, shuffle=True)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=8, shuffle=False)

train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=8, shuffle=True)
test_loader_roberta = DataLoader(test_dataset_roberta, batch_size=8, shuffle=False)

train_loader_deberta = DataLoader(train_dataset_deberta, batch_size=8, shuffle=True)
test_loader_deberta = DataLoader(test_dataset_deberta, batch_size=8, shuffle=False)

# Load transformer models
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
deberta_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)

# Move models to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bert_model.to(device)
roberta_model.to(device)
deberta_model.to(device)

# Define optimizer & scheduler
optimizer_bert = AdamW(bert_model.parameters(), lr=3e-5)
optimizer_roberta = AdamW(roberta_model.parameters(), lr=3e-5)
optimizer_deberta = AdamW(deberta_model.parameters(), lr=3e-5)

num_training_steps = len(train_loader_bert) * 3  # 3 epochs
lr_scheduler_bert = get_scheduler("linear", optimizer=optimizer_bert, num_warmup_steps=0, num_training_steps=num_training_steps)
lr_scheduler_roberta = get_scheduler("linear", optimizer=optimizer_roberta, num_warmup_steps=0, num_training_steps=num_training_steps)
lr_scheduler_deberta = get_scheduler("linear", optimizer=optimizer_deberta, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training function
def train_model(model, train_loader, optimizer, scheduler, epochs=3):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

# Train all models
train_model(bert_model, train_loader_bert, optimizer_bert, lr_scheduler_bert)
train_model(roberta_model, train_loader_roberta, optimizer_roberta, lr_scheduler_roberta)
train_model(deberta_model, train_loader_deberta, optimizer_deberta, lr_scheduler_deberta)

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels)
    return predictions, true_labels

# Get predictions from all three models
bert_preds, true_labels = evaluate_model(bert_model, test_loader_bert)
roberta_preds, _ = evaluate_model(roberta_model, test_loader_roberta)
deberta_preds, _ = evaluate_model(deberta_model, test_loader_deberta)

# Ensemble Model - Majority Voting
final_preds = np.round((np.array(bert_preds) + np.array(roberta_preds) + np.array(deberta_preds)) / 3).astype(int)

# Compute Accuracy & Classification Report
accuracy = accuracy_score(true_labels, final_preds)
report = classification_report(true_labels, final_preds, target_names=["Non-Rumor", "Rumor"])

print(f"Hybrid Model Test Accuracy: {accuracy:.4f}")
print(report)
