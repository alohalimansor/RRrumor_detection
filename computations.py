import numpy as np
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Model and Tokenizer Setup
# -----------------------------
model_path = "./fine_tuned_roberta_augmented"  # Path to the fine-tuned RoBERTa model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode


# -----------------------------
# Prediction Function
# -----------------------------
def predict_proba(texts):
    """
    Convert text inputs to probability outputs

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


# -----------------------------
# Define SubwordMasker
# -----------------------------
class SubwordMasker:
    """
    Class to mask tokens for SHAP analysis
    """

    def __init__(self, tokenizer, mask_token="[UNK]"):
        self.tokenizer = tokenizer
        self.mask_token = mask_token

    def __call__(self, text, mask):
        tokens = self.tokenizer.tokenize(text)
        if len(mask) > len(tokens):
            mask = mask[:len(tokens)]
        masked_tokens = [t if mask[i] else self.mask_token for i, t in enumerate(tokens)]
        return self.tokenizer.convert_tokens_to_string(masked_tokens)


# -----------------------------
# Compute SHAP values for a given text
# -----------------------------
def compute_shap_values(text, tokenizer, predict_proba, masker, nsamples=200):
    """
    Compute SHAP values for tokens in a text

    Args:
        text: Input text to explain
        tokenizer: Tokenizer for the model
        predict_proba: Prediction function
        masker: SubwordMasker instance
        nsamples: Number of samples for SHAP computation

    Returns:
        tokens: List of tokens from the text
        shap_values: SHAP values for each token
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) == 0:
        return tokens, np.array([])
    background = np.ones((1, len(tokens)))
    explainer = shap.KernelExplainer(
        lambda masks: predict_proba([masker(text, mask) for mask in masks])[:, 1],
        background,
        nsamples=nsamples
    )
    shap_values = explainer.shap_values(background)[0]
    return tokens, shap_values


# -----------------------------
# Evaluate Fidelity
# -----------------------------
def evaluate_fidelity(text, tokenizer, predict_proba, masker, top_fraction=0.2):
    """
    Evaluate explanation fidelity by measuring prediction change when masking important tokens

    Args:
        text: Input text
        tokenizer: Tokenizer for the model
        predict_proba: Prediction function
        masker: SubwordMasker instance
        top_fraction: Fraction of tokens to mask (by importance)

    Returns:
        fidelity: Change in probability after masking
        tokens_to_remove: List of tokens that were masked
        p_orig: Original prediction probability
        p_new: Prediction probability after masking
    """
    tokens, shap_values = compute_shap_values(text, tokenizer, predict_proba, masker)
    if len(tokens) == 0 or shap_values.size == 0:
        return None, [], None, None

    p_orig = predict_proba(text)[0][1]  # Target class probability (e.g., rumor)
    num_tokens = len(tokens)
    num_remove = max(1, int(num_tokens * top_fraction))

    # Pair tokens with their SHAP values and sort by importance (absolute value)
    token_importances = list(zip(tokens, shap_values))
    token_importances.sort(key=lambda x: abs(x[1]), reverse=True)
    tokens_to_remove = [t for t, _ in token_importances[:num_remove]]

    # Create new text with selected tokens masked
    new_tokens = [token if token not in tokens_to_remove else "[UNK]" for token in tokens]
    new_text = " ".join(new_tokens)
    p_new = predict_proba(new_text)[0][1]

    fidelity = p_orig - p_new
    return fidelity, tokens_to_remove, p_orig, p_new


# -----------------------------
# Text Perturbation Function
# -----------------------------
def perturb_text(text):
    """
    Creates a small perturbation by swapping two adjacent words

    Args:
        text: Original text

    Returns:
        Perturbed text with one pair of adjacent words swapped
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text
    idx = random.randint(0, len(tokens) - 2)
    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
    return " ".join(tokens)


# -----------------------------
# Extract Top Features
# -----------------------------
def extract_top_features(text, tokenizer, predict_proba, masker, top_k=3, nsamples=200):
    """
    Extract the top k tokens by SHAP value importance

    Args:
        text: Input text
        tokenizer: Tokenizer for the model
        predict_proba: Prediction function
        masker: SubwordMasker instance
        top_k: Number of top features to extract
        nsamples: Number of samples for SHAP computation

    Returns:
        Set of top k tokens by importance
    """
    tokens, shap_values = compute_shap_values(text, tokenizer, predict_proba, masker, nsamples=nsamples)
    if len(tokens) == 0 or shap_values.size == 0:
        return set()
    token_importances = list(zip(tokens, shap_values))
    token_importances.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [t for t, _ in token_importances[:top_k]]
    return set(top_features)


# -----------------------------
# Evaluate Stability
# -----------------------------
def evaluate_stability(text, tokenizer, predict_proba, masker, num_perturbations=5, top_k=3, nsamples=200):
    """
    Evaluate the stability of explanations under text perturbations

    Args:
        text: Input text
        tokenizer: Tokenizer for the model
        predict_proba: Prediction function
        masker: SubwordMasker instance
        num_perturbations: Number of perturbed versions to generate
        top_k: Number of top features to consider
        nsamples: Number of samples for SHAP computation

    Returns:
        Average Jaccard similarity between original and perturbed explanations
    """
    original_features = extract_top_features(text, tokenizer, predict_proba, masker, top_k, nsamples)
    similarities = []
    for _ in range(num_perturbations):
        perturbed_text = perturb_text(text)
        perturbed_features = extract_top_features(perturbed_text, tokenizer, predict_proba, masker, top_k, nsamples)
        if not original_features and not perturbed_features:
            similarity = 1.0
        else:
            intersection = original_features.intersection(perturbed_features)
            union = original_features.union(perturbed_features)
            similarity = len(intersection) / len(union)
        similarities.append(similarity)
    stability_score = sum(similarities) / len(similarities)
    return stability_score


# -----------------------------
# Evaluate Sparsity
# -----------------------------
def evaluate_sparsity(text, tokenizer, predict_proba, masker, threshold=0.005, nsamples=200):
    """
    Evaluate explanation sparsity as fraction of tokens with significant SHAP values

    Args:
        text: Input text
        tokenizer: Tokenizer for the model
        predict_proba: Prediction function
        masker: SubwordMasker instance
        threshold: SHAP value threshold for significance
        nsamples: Number of samples for SHAP computation

    Returns:
        Sparsity score (lower is more sparse/focused)
    """
    tokens, shap_values = compute_shap_values(text, tokenizer, predict_proba, masker, nsamples=nsamples)
    if len(tokens) == 0:
        return 0.0
    num_significant = sum(1 for v in shap_values if abs(v) > threshold)
    sparsity = num_significant / len(tokens)
    return sparsity


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import random

    masker = SubwordMasker(tokenizer)

    # Sample text for demonstration
    sample_text = "Charlie Hebdo became well known for publishing the Muhammed cartoons two years ago"

    # Evaluate Fidelity
    fidelity, removed_tokens, p_orig, p_new = evaluate_fidelity(sample_text, tokenizer, predict_proba, masker,
                                                                top_fraction=0.2)
    fidelity_abs = abs(p_orig - p_new)

    print("Fidelity Evaluation for Sample Text:")
    print("Original Prediction (rumor probability):", p_orig)
    print("New Prediction after masking tokens {}: {}".format(removed_tokens, p_new))
    print("Fidelity (drop in probability):", fidelity)
    print("Absolute Fidelity (change in probability):", fidelity_abs)

    # Evaluate Stability
    stability_score = evaluate_stability(sample_text, tokenizer, predict_proba, masker, num_perturbations=5, top_k=3,
                                         nsamples=200)
    print("Stability Score (average Jaccard similarity):", stability_score)

    # Evaluate Sparsity
    sparsity_score = evaluate_sparsity(sample_text, tokenizer, predict_proba, masker, threshold=0.005, nsamples=200)
    print("Sparsity Score (fraction of tokens with significant importance):", sparsity_score)