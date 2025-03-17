import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Define PyTorch Dataset
class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Ensure this is moved to the right device later
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        device = logits.device  # Get the device of the input tensor
        self.alpha = self.alpha.to(device)  # Move class weights to the same device

        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# Define Custom Trainer (Fixed `compute_loss`)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # **kwargs ensures compatibility
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = FocalLoss(alpha=class_weights, gamma=2.0)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Global variable for class weights
class_weights = None

def main():
    # Parse arguments for dynamic batch size
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()

    # Verify GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load the augmented dataset
    file_path = "augmented_twitter_rumor_dataset.csv"
    df = pd.read_csv(file_path, encoding="utf-8")

    # Drop any missing values
    df = df.dropna(subset=["text", "label"])
    print(f"Total samples after dropping nulls: {len(df)}")

    # Compute class weights
    global class_weights
    class_weights = compute_class_weight("balanced", classes=np.unique(df["label"]), y=df["label"])
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights: {class_weights}")

    # Split dataset
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(), df["label"].astype(int).tolist(), test_test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./pretrained_roberta_mlm")

    # Create dataset objects
    train_dataset = RumorDataset(train_texts, train_labels, tokenizer)
    test_dataset = RumorDataset(test_texts, test_labels, tokenizer)

    # Load the pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained("./pretrained_roberta_mlm", num_labels=2)

    # Move model & weights to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights_to_device = class_weights.to(device)  # Local copy for trainer

    # Define Training Arguments (Fixed `eval_strategy`)
    training_args = TrainingArguments(
        output_dir="./fine_tuned_roberta_augmented",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",  # Fixed the deprecated `evaluation_strategy`
        save_total_limit=2,
        logging_dir="./logs",
        learning_rate=2e-5,
        fp16=True,
        dataloader_num_workers=4,
    )

    # Define Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_roberta_augmented")
    tokenizer.save_pretrained("./fine_tuned_roberta_augmented")

    # Save Training Logs
    with open("training_log.txt", "a") as log_file:
        log_file.write(f"Training completed on device: {device}\n")
        log_file.write(f"Final model saved at: fine_tuned_roberta_augmented\n")

    print("\nFine-Tuning on Augmented Data Completed! Model saved as 'fine_tuned_roberta_augmented'.")

if __name__ == "__main__":
    main()