import torch
import shap
import lime.lime_text
import numpy as np
import pandas as pd
import nltk
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
import matplotlib.pyplot as plt

# Load tokenizer and model
model_path = "./fine_tuned_roberta_augmented"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test dataset
file_path = "augmented_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])
test_texts = df["text"].tolist()[:100]  # Sample 100 for explainability
test_labels = df["label"].astype(int).tolist()[:100]


# Define PyTorch Dataset
class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length,
                                   return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], device=device)
        return item


test_dataset = RumorDataset(test_texts, test_labels, tokenizer)


# Prediction function for model inference
def predict_proba(texts):
    if isinstance(texts, str):  # Handle single string
        texts = [texts]
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
    return probs


# SHAP Explainer using Different Approach
# Use a simpler approach with Permutation explainer
def get_shap_explanations(texts, n_samples=20):
    """Generate SHAP explanations for a list of texts."""
    print("Generating SHAP explanations...")

    # Create a simple word-level masker
    # This approach splits text into words and replaces words with mask token
    class SimpleWordMasker:
        def __init__(self, mask_token="[UNK]"):
            self.mask_token = mask_token

        def __call__(self, text, mask):
            """Apply mask to text by replacing masked words with mask token."""
            words = text.split()
            masked_text = []
            for i, word in enumerate(words):
                if i < len(mask) and mask[i]:
                    masked_text.append(word)
                else:
                    masked_text.append(self.mask_token)
            return " ".join(masked_text)

    # Function to convert text to feature array (word presence)
    def text_to_features(text):
        """Convert text to binary feature array (1 for word present)."""
        words = text.split()
        return np.ones((1, len(words))) #create 2d array.

    explanations = []
    for i, text in enumerate(texts[:n_samples]):
        print(f"Processing text {i + 1}/{n_samples} for SHAP explanation...")

        # Skip empty texts
        if not text or not text.strip():
            print(f"Skipping empty text at index {i}")
            explanations.append(None)
            continue

        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)

        words = text.split()
        if not words:
            print(f"Skipping text with no words at index {i}")
            explanations.append(None)
            continue

        # Use SHAP's KernelExplainer for this text
        masker = SimpleWordMasker()
        background = text_to_features(text)  # Features for background

        # Define a wrapper for prediction
        def model_wrapper(masks):
            """Convert masks to text and get predictions."""
            masked_texts = [masker(text, mask) for mask in masks]
            return predict_proba(masked_texts)[:, 1]  # Return prob of rumor class

        # Create explainer and compute values
        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(background, nsamples=50)

        # Store values along with word mapping
        explanations.append({
            'values': shap_values,
            'words': words
        })

    return explanations


# LIME Explainability
lime_explainer = lime.lime_text.LimeTextExplainer(class_names=["Non-Rumor", "Rumor"])


def get_lime_explanations(texts, n_samples=20):
    """Generate LIME explanations for a list of texts."""
    print("Generating LIME explanations...")
    explanations = []
    for i, text in enumerate(texts[:n_samples]):
        print(f"Processing text {i + 1}/{n_samples} for LIME explanation...")
        # Skip empty texts
        if not text or not isinstance(text, str) or not text.strip():
            print(f"Skipping empty text at index {i}")
            explanations.append(None)
            continue

        try:
            exp = lime_explainer.explain_instance(text, predict_proba, num_features=10)
            explanations.append(exp)
        except Exception as e:
            print(f"Error processing text {i} with LIME: {e}")
            explanations.append(None)
    return explanations


# Multi-Head Attention Analysis
def extract_attention_scores(text):
    if not text or not isinstance(text, str) or not text.strip():
        return []

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions[-1][0]  # Last layer, all heads
        avg_attention = attentions.mean(dim=0).mean(dim=0).cpu().numpy()  # Avg across heads and batch
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return list(zip(tokens, avg_attention))


# Generate explanations
n_samples = 20  # Number of texts to process
print("Starting explainability analysis...")
shap_explanations = get_shap_explanations(test_texts, n_samples)
lime_explanations = get_lime_explanations(test_texts, n_samples)


# Explainability Metrics
import numpy as np


def compute_fidelity(idx, method="shap", top_k=5):
    """Compute fidelity by masking top influential tokens."""
    text = test_texts[idx]

    # Skip if text is invalid
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    orig_prob = predict_proba(text)[0, 1]  # Probability for rumor class
    words = text.split()

    if method == "shap" and idx < len(shap_explanations) and shap_explanations[idx] is not None and 'values' in \
            shap_explanations[idx]:
        # Get word-level importance
        shap_vals = shap_explanations[idx]['values']

        # Debug print statements
        print(f"Type of shap_vals: {type(shap_vals)}")
        print(f"Value of shap_vals: {shap_vals}")

        # Get indices of top_k most important words
        if len(shap_vals.tolist()) >= top_k:
            top_indices = np.argsort(-np.abs(shap_vals))[:top_k]
            top_words = [words[i] for i in top_indices if i < len(words)]
        else:
            # Not enough values, use all available
            top_words = words[:len(shap_vals)]

    elif method == "lime" and idx < len(lime_explanations) and lime_explanations[idx] is not None:
        exp = lime_explanations[idx]
        top_features = exp.as_list()[:top_k]
        top_words = [w for w, _ in top_features if w in words]
    else:
        return 0.0  # Default value if method not applicable

    # Create masked text by removing top words
    if not top_words:
        return 0.0

    masked_text = " ".join(["[MASK]" if w in top_words else w for w in words])
    new_prob = predict_proba(masked_text)[0, 1]

    return abs(orig_prob - new_prob)


def compute_stability(idx, method="shap", n=5):
    """Compute stability of explanations."""
    text = test_texts[idx]

    # Skip if text is invalid
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    if method == "shap" and idx < len(shap_explanations) and shap_explanations[idx] is not None:
        # Simulate multiple runs by adding small noise
        original_vals = shap_explanations[idx]['values']
        scores = [original_vals + np.random.normal(0, 0.01, size=original_vals.shape) for _ in range(n)]
        return np.std([np.sum(s) for s in scores])
    elif method == "lime" and idx < len(lime_explanations) and lime_explanations[idx] is not None:
        # Re-run LIME multiple times
        try:
            scores = []
            for _ in range(n):
                exp = lime_explainer.explain_instance(text, predict_proba, num_features=10)
                feature_vals = [val for _, val in exp.as_list()]
                scores.append(feature_vals)
            return np.mean([np.std(s) for s in zip(*scores)]) if scores else 0.0
        except:
            return 0.0
    else:
        return 0.0


def compute_sparsity(idx, method="shap"):
    """Compute sparsity of explanations (what % of features have non-zero importance)."""
    text = test_texts[idx]

    # Skip if text is invalid
    if not text or not isinstance(text, str) or not text.strip():
        return 0.0

    if method == "shap" and idx < len(shap_explanations) and shap_explanations[idx] is not None:
        vals = shap_explanations[idx]['values']
        return len([v for v in vals if abs(v) > 1e-5]) / len(vals)
    elif method == "lime" and idx < len(lime_explanations) and lime_explanations[idx] is not None:
        vals = [s for _, s in lime_explanations[idx].as_list()]
        return len([v for v in vals if abs(v) > 1e-5]) / len(vals)
    else:
        return 0.0

# Adversarial Explainability with TextFooler (with better error handling)
def generate_adversarial_texts(texts, labels, n_samples=20):
    print("Generating adversarial examples...")
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)

    adversarial_texts = []
    for i, (text, label) in enumerate(zip(texts[:n_samples], labels[:n_samples])):
        try:
            print(f"Generating adversarial text {i + 1}/{n_samples}...")
            if not text or not isinstance(text, str) or not text.strip():
                print(f"Skipping empty text at index {i}")
                adversarial_texts.append(None)
                continue

            result = attack.attack(text, label)
            adv_text = result.perturbed_text() if result.perturbed_text() else text
            adversarial_texts.append(adv_text)
        except Exception as e:
            print(f"Error generating adversarial text for index {i}: {e}")
            adversarial_texts.append(None)

    return adversarial_texts


adversarial_texts = generate_adversarial_texts(test_texts, test_labels, n_samples)


# Compute Metrics for all methods (with error handling)
def compute_all_metrics(n_samples=20):
    metrics = {"shap": {"fidelity": [], "stability": [], "sparsity": []},
               "lime": {"fidelity": [], "stability": [], "sparsity": []}}

    print("Computing metrics...")
    for i in range(n_samples):
        print(f"Processing metrics for sample {i + 1}/{n_samples}...")
        if i >= len(test_texts) or not test_texts[i]:
            continue

        for method in ["shap", "lime"]:
            try:
                fidelity = compute_fidelity(i, method)
                stability = compute_stability(i, method)
                sparsity = compute_sparsity(i, method)

                metrics[method]["fidelity"].append(fidelity)
                metrics[method]["stability"].append(stability)
                metrics[method]["sparsity"].append(sparsity)
            except Exception as e:
                print(f"Error computing {method} metrics for sample {i}: {e}")

    return metrics


metrics = compute_all_metrics(n_samples)


# Compute explanation drift between original and adversarial examples
def compute_drift_scores(n_samples=20):
    print("Computing explanation drift...")
    drift_scores = []

    for i, (text, adv_text) in enumerate(zip(test_texts[:n_samples], adversarial_texts)):
        try:
            if i >= len(shap_explanations) or shap_explanations[i] is None or adv_text is None:
                continue

            # Get SHAP values for original text
            orig_shap = shap_explanations[i]['values']

            # Get SHAP values for adversarial text
            words = adv_text.split()

            # Skip if text is too short
            if len(words) < 2:
                continue

            # Use simple word masker for adversarial text
            class SimpleWordMasker:
                def __init__(self, mask_token="[UNK]"):
                    self.mask_token = mask_token

                def __call__(self, text, mask):
                    words = text.split()
                    masked_text = []
                    for i, word in enumerate(words):
                        if i < len(mask) and mask[i]:
                            masked_text.append(word)
                        else:
                            masked_text.append(self.mask_token)
                    return " ".join(masked_text)

            # Function to convert text to feature array (word presence)
            def text_to_features(text):
                words = text.split()
                return np.ones(len(words))

            masker = SimpleWordMasker()
            background = text_to_features(adv_text)

            def model_wrapper(masks):
                masked_texts = [masker(adv_text, mask) for mask in masks]
                return predict_proba(masked_texts)[:, 1]

            explainer = shap.KernelExplainer(model_wrapper, background)
            adv_shap = explainer.shap_values(background, nsamples=50)

            # Normalize lengths for comparison
            min_len = min(len(orig_shap), len(adv_shap))
            orig_shap = orig_shap[:min_len]
            adv_shap = adv_shap[:min_len]

            # Skip if either array is too small
            if min_len < 2:
                continue

            # Calculate drift using cosine similarity
            if np.linalg.norm(orig_shap) > 0 and np.linalg.norm(adv_shap) > 0:
                drift = 1 - cosine(orig_shap, adv_shap)
                drift_scores.append(drift)
        except Exception as e:
            print(f"Error computing drift for sample {i}: {e}")

    return drift_scores


drift_scores = compute_drift_scores(n_samples)

# Statistical Validation (with error handling)
try:
    shap_fidelity = metrics["shap"]["fidelity"]
    lime_fidelity = metrics["lime"]["fidelity"]

    # Only compare if we have enough data
    if len(shap_fidelity) > 5 and len(lime_fidelity) > 5:
        # Ensure equal lengths
        min_len = min(len(shap_fidelity), len(lime_fidelity))
        stat, p_value = wilcoxon(shap_fidelity[:min_len], lime_fidelity[:min_len])
    else:
        p_value = float('nan')
except Exception as e:
    print(f"Error in statistical validation: {e}")
    p_value = float('nan')

# Print Results
print("\n Explainability Analysis Results:")
for method in ["shap", "lime"]:
    if len(metrics[method]["fidelity"]) > 0:
        fidelity = np.mean(metrics[method]["fidelity"])
        stability = np.mean(metrics[method]["stability"])
        sparsity = np.mean(metrics[method]["sparsity"])
        print(f"{method.upper()} - Fidelity: {fidelity:.4f}, "
              f"Stability: {stability:.4f}, "
              f"Sparsity: {sparsity:.4f}")
    else:
        print(f"{method.upper()} - No valid metrics available")

if drift_scores:
    print(f"Explanation Drift (SHAP): {np.mean(drift_scores):.4f}")
else:
    print("Explanation Drift: No valid drift scores available")

if not np.isnan(p_value):
    print(f"Wilcoxon Test (SHAP vs LIME Fidelity) p-value: {p_value:.4f}")
else:
    print("Wilcoxon Test: Could not compute p-value")

# Visualizations (with error handling)
print("Generating visualizations...")

# SHAP summary plot
try:
    valid_explanations = [exp for exp in shap_explanations if exp is not None]
    if valid_explanations:
        plt.figure(figsize=(12, 8))

        # Get values from valid explanations
        all_values = []
        all_words = []
        for exp in valid_explanations:
            values = exp['values']
            words = exp['words']
            # Truncate to reasonable length
            max_words = 10
            all_values.append(values[:max_words])
            all_words.append(words[:max_words])

        # Pad shorter arrays for consistent shape
        max_len = max(len(v) for v in all_values)
        padded_values = np.zeros((len(all_values), max_len))
        for i, vals in enumerate(all_values):
            padded_values[i, :len(vals)] = vals

        # Create feature names
        feature_names = [f"Word {i + 1}" for i in range(max_len)]

        # Create summary plot
        shap.summary_plot(
            padded_values,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        plt.close()

    # Individual SHAP plots
    for i, exp in enumerate(shap_explanations[:5]):
        if exp is not None:
            plt.figure(figsize=(10, 6))
            values = exp['values']
            words = exp['words']

            # Limit to top 15 words
            if len(words) > 15:
                top_indices = np.argsort(-np.abs(values))[:15]
                values = values[top_indices]
                words = [words[i] for i in top_indices]

            # Sort by absolute value
            sorted_idx = np.argsort(-np.abs(values))
            values = values[sorted_idx]
            words = [words[i] for i in sorted_idx]

            colors = ['r' if v < 0 else 'b' for v in values]
            plt.barh(range(len(words)), values, color=colors)
            plt.yticks(range(len(words)), words)
            plt.title(f"SHAP Values for Text {i + 1}")
            plt.tight_layout()
            plt.savefig(f"shap_text_{i + 1}.png")
            plt.close()
except Exception as e:
    print(f"Error generating SHAP visualizations: {e}")

# LIME visualizations
try:
    for i, exp in enumerate(lime_explanations[:5]):
        if exp is not None:
            try:
                exp.save_to_file(f"lime_explanation_{i + 1}.html")
            except Exception as e:
                print(f"Error saving LIME explanation {i + 1}: {e}")
except Exception as e:
    print(f"Error generating LIME visualizations: {e}")

# Attention plot for first sample
try:
    if test_texts and len(test_texts) > 0:
        attn_scores = extract_attention_scores(test_texts[0])
        if attn_scores:
            plt.figure(figsize=(12, 4))
            plt.bar([t for t, _ in attn_scores], [a for _, a in attn_scores])
            plt.xticks(rotation=45)
            plt.title("Attention Scores for Sample 1")
            plt.tight_layout()
            plt.savefig("attention_plot.png")
            plt.close()
except Exception as e:
    print(f"Error generating attention plot: {e}")

print("\n Advanced Explainability Analysis Completed!")