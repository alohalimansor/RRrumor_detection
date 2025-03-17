import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# Load the fine-tuned RoBERTa model and tokenizer
model_path = "./fine_tuned_roberta_augmented"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode


# Define prediction function for LIME explainer
def predict_proba(texts):
    """
    Convert text inputs to probability outputs for LIME

    Args:
        texts: String or list of strings to classify

    Returns:
        numpy array of probabilities for each class
    """
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    return probs


# Load test data from the augmented Twitter rumor dataset
file_path = "augmented_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])
test_texts = df["text"].tolist()[:100]  # Use first 100 samples as specified in the paper
test_labels = df["label"].astype(int).tolist()[:100]


# Function to create simple text perturbations for stability testing
def perturb_text(text):
    """
    Create a perturbed version of the text by swapping two adjacent tokens

    Args:
        text: Original text string

    Returns:
        Perturbed text with one pair of adjacent words swapped
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text
    idx = random.randint(0, len(tokens) - 2)
    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
    return " ".join(tokens)


# Extract top features using LIME
def extract_top_features_lime(text, top_k=3, num_samples=200):
    """
    Extract the top k features from LIME explanation

    Args:
        text: Text to explain
        top_k: Number of top features to extract
        num_samples: Number of samples for LIME

    Returns:
        Set of top k features
    """
    explainer = LimeTextExplainer(class_names=["Not Rumor", "Rumor"])
    exp = explainer.explain_instance(text, predict_proba, num_features=10, num_samples=num_samples)
    top_features = [x[0] for x in exp.as_list()[:top_k]]
    return set(top_features)


# Evaluate explanation stability for LIME
def evaluate_stability_lime(text, num_perturbations=5, top_k=3, num_samples=200):
    """
    Evaluate the stability of LIME explanations under text perturbations

    Args:
        text: Original text
        num_perturbations: Number of perturbed versions to generate
        top_k: Number of top features to consider
        num_samples: Number of samples for LIME

    Returns:
        Average Jaccard similarity between original and perturbed explanations
    """
    original_features = extract_top_features_lime(text, top_k, num_samples)
    similarities = []
    for _ in range(num_perturbations):
        perturbed_text = perturb_text(text)
        perturbed_features = extract_top_features_lime(perturbed_text, top_k, num_samples)
        if not original_features and not perturbed_features:
            similarity = 1.0
        else:
            intersection = original_features.intersection(perturbed_features)
            union = original_features.union(perturbed_features)
            similarity = len(intersection) / len(union) if union else 1.0
        similarities.append(similarity)
    return sum(similarities) / len(similarities)


# Compute LIME stability for all test samples
lime_stabilities = []
for text in test_texts:
    stability_score = evaluate_stability_lime(text, num_perturbations=5, top_k=3, num_samples=200)
    lime_stabilities.append(stability_score)

lime_stability = np.mean(lime_stabilities)

# Values from the paper for STEF and SHAP (for comparison)
stef_stability = 0.742  # From the paper (section 4.2)
shap_stability = 0.613  # From the paper (section 4.2)

# Create visualization comparing the three methods
methods = ['STEF', 'SHAP', 'LIME']
stabilities = [stef_stability, shap_stability, lime_stability]
plt.figure(figsize=(8, 6))
bars = plt.bar(methods, stabilities, color=['#5F9EA0', '#FF9999', '#66B2FF'])
plt.title('Comparison of Explanation Stability (Jaccard Similarity)')
plt.ylabel('Stability Score')
plt.ylim(0, 1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom')
plt.savefig('stability_comparison.png')
plt.show()

print(f"LIME Stability: {lime_stability:.3f}")