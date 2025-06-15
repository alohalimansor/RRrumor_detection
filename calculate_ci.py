import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = r"./fine_tuned_roberta_augmented"
DATA_FILE_PATH = "augmented_twitter_rumor_dataset.csv"

# Bootstrap configuration
N_BOOTSTRAPS = 2000
CONFIDENCE_LEVEL = 0.95


def diagnostic_with_confidence_intervals():
    """
    DIAGNOSTIC SCRIPT: Evaluates the model on the ENTIRE dataset and calculates
    confidence intervals to check if this reproduces the inflated high score.
    """
    # 1. Load Model and Tokenizer
    print(f"Loading model from: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load the FULL dataset for evaluation (NO TRAIN-TEST SPLIT)
    print("Loading data for evaluation...")
    df = pd.read_csv(DATA_FILE_PATH, encoding="utf-8").dropna(subset=["text", "label"])

    true_labels = df["label"].astype(int).values
    texts = df["text"].tolist()
    print(f"DIAGNOSTIC: Evaluating on the FULL dataset with {len(texts)} samples (includes training data).")

    # 3. Generate Predictions for the full dataset (run once)
    all_predictions = []
    print("Generating predictions on the full dataset...")
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        all_predictions.append(predicted_class_id)

    all_predictions = np.array(all_predictions)

    # 4. Bootstrap Confidence Intervals
    print(f"\nStarting bootstrapping with {N_BOOTSTRAPS} iterations...")
    n_samples = len(true_labels)
    bootstrapped_accuracy = []
    bootstrapped_f1 = []
    bootstrapped_precision = []
    bootstrapped_recall = []

    for _ in tqdm(range(N_BOOTSTRAPS)):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        y_true_sample = true_labels[indices]
        y_pred_sample = all_predictions[indices]

        bootstrapped_accuracy.append(accuracy_score(y_true_sample, y_pred_sample))
        bootstrapped_f1.append(f1_score(y_true_sample, y_pred_sample, zero_division=0))
        bootstrapped_precision.append(precision_score(y_true_sample, y_pred_sample, zero_division=0))
        bootstrapped_recall.append(recall_score(y_true_sample, y_pred_sample, zero_division=0))

    # 5. Calculate and Print Final Report
    print("\n--- DIAGNOSTIC Performance Report (on Full Dataset) ---")

    lower_percentile = (1.0 - CONFIDENCE_LEVEL) / 2.0 * 100
    upper_percentile = (CONFIDENCE_LEVEL + (1.0 - CONFIDENCE_LEVEL) / 2.0) * 100

    def print_metric_ci(metric_name, scores, original_metric):
        lower_bound = np.percentile(scores, lower_percentile)
        upper_bound = np.percentile(scores, upper_percentile)
        print(f"\n{metric_name}:")
        print(f"  - Score on Full Dataset: {original_metric:.4f}")
        print(f"  - 95% CI: ({lower_bound:.4f} - {upper_bound:.4f})")

    original_accuracy = accuracy_score(true_labels, all_predictions)
    original_f1 = f1_score(true_labels, all_predictions)
    original_precision = precision_score(true_labels, all_predictions)
    original_recall = recall_score(true_labels, all_predictions)

    print_metric_ci("Accuracy", bootstrapped_accuracy, original_accuracy)
    print_metric_ci("F1-Score", bootstrapped_f1, original_f1)
    print_metric_ci("Precision", bootstrapped_precision, original_precision)
    print_metric_ci("Recall", bootstrapped_recall, original_recall)

    print("\n-----------------------------------------------------")


if __name__ == '__main__':
    diagnostic_with_confidence_intervals()