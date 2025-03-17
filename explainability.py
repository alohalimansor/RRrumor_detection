import torch
import shap
import numpy as np
import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper

# -----------------------------
# Model and Tokenizer Setup
# -----------------------------
model_path = "./fine_tuned_roberta_augmented"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -----------------------------
# Data Loading
# -----------------------------
file_path = "augmented_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])
test_texts = [
    str(t) if pd.notna(t) and isinstance(t, (str, float, np.number)) else ""
    for t in df["text"].tolist()[:100]
]
test_labels = df["label"].astype(int).tolist()[:100]


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
# Subword-Aware SHAP Explanations
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


def get_shap_explanations(texts, n_samples=20):
    """
    Generate SHAP-based explanations for a list of texts

    Args:
        texts: List of text strings to explain
        n_samples: Number of samples to process

    Returns:
        List of structured explanations based on SHAP values
    """
    explanations = []
    masker = SubwordMasker(tokenizer)
    for text in texts[:n_samples]:
        if not text.strip():
            explanations.append(None)
            continue
        try:
            tokens = tokenizer.tokenize(text)
            background = np.ones((1, len(tokens)))
            explainer = shap.KernelExplainer(
                lambda masks: predict_proba([masker(text, mask) for mask in masks])[:, 1],
                background,
                nsamples=200  # Increased for better stability
            )
            shap_values = explainer.shap_values(background)[0]
            words = tokenizer.convert_tokens_to_string(tokens).split()
            # Debug: Print SHAP values
            print(f"SHAP values for '{text}': {list(zip(words, shap_values[:len(words)]))}")
            summary = categorize_and_summarize(words, shap_values)
            explanations.append(summary)
        except Exception as e:
            print(f"Error in SHAP: {e}")
            explanations.append(None)
    return explanations


def categorize_and_summarize(words, shap_values, threshold=0.005):
    """
    Categorize words based on their SHAP values and semantic categories

    Args:
        words: List of words from the text
        shap_values: SHAP values for each word
        threshold: Minimum absolute SHAP value to consider significant

    Returns:
        Formatted summary of the explanation
    """
    emotion_words = set(["urgent", "shocking", "dead", "scary", "amazing", "fear"])
    vague_words = set(["some", "maybe", "rumor", "people", "guessing", "considered"])
    authority_words = set(["expert", "official", "government", "police", "hebdo"])
    event_words = set(["shooting", "terrorism", "cartoons", "today", "dead", "publishing"])
    summary = {"emotion": [], "vague": [], "authority": [], "event": [], "other": []}
    for word, value in zip(words, shap_values[:len(words)]):
        if abs(value) < threshold:
            continue
        word_lower = word.lower()
        if word_lower in emotion_words:
            summary["emotion"].append((word, value))
        elif word_lower in vague_words:
            summary["vague"].append((word, value))
        elif word_lower in authority_words:
            summary["authority"].append((word, value))
        elif word_lower in event_words:
            summary["event"].append((word, value))
        else:
            summary["other"].append((word, value))
    return format_summary(summary)


def format_summary(summary):
    """
    Format the categorized summary into a human-readable explanation

    Args:
        summary: Dictionary of categorized words and their SHAP values

    Returns:
        String containing the formatted explanation
    """
    text = []
    if summary["emotion"]:
        text.append(f"Emotional words like {', '.join(w for w, _ in summary['emotion'])} suggest a rumor.")
    if summary["vague"]:
        text.append(f"Vague terms like {', '.join(w for w, _ in summary['vague'])} make it less reliable.")
    if summary["authority"]:
        text.append(f"Authority mentions like {', '.join(w for w, _ in summary['authority'])} affect the prediction.")
    if summary["event"]:
        text.append(f"Events like {', '.join(w for w, _ in summary['event'])} caught the model's attention.")
    if not any(summary.values()):
        text.append("No strong clues found—prediction based on overall tone.")
    return " ".join(text)


# -----------------------------
# Simplified Attention Extraction
# -----------------------------
def extract_attention_scores(text):
    """
    Extract and analyze attention scores from the model

    Args:
        text: Input text to analyze

    Returns:
        String describing which words the model focused on
    """
    if not isinstance(text, str) or not text.strip():
        return "No focus identified."
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions[-1].mean(dim=1).squeeze(0).mean(dim=0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    words = tokenizer.convert_tokens_to_string(tokens).split()
    # Enhanced stopword filter
    stopwords = set(
        ["the", "a", "an", "is", "are", "was", "and", "or", "in", "on", "at", "to", ".", ",", ":", "'", "?", "<s>",
         "</s>"]
    )
    valid_pairs = [(w, attentions[i]) for i, w in enumerate(words) if w.lower() not in stopwords and len(w) > 2]
    if not valid_pairs:
        return "No clear focus."
    top_words = sorted(valid_pairs, key=lambda x: x[1], reverse=True)[:3]
    return f"The model focused on: {', '.join(w for w, _ in top_words)}."


# -----------------------------
# Adversarial Trust Test
# -----------------------------
def generate_adversarial_texts(texts, labels, n_samples=20):
    """
    Generate adversarial examples and evaluate model robustness

    Args:
        texts: List of original texts
        labels: List of original labels
        n_samples: Number of samples to process

    Returns:
        List of trust assessments based on adversarial testing
    """
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)
    results = []

    for text, label in zip(texts[:n_samples], labels[:n_samples]):
        if not text.strip():
            results.append(None)
            continue
        try:
            result = attack.attack(text, label)
            adv_text = result.perturbed_text() if result.perturbed_text() else text
            orig_prob = predict_proba([text])[0][1]
            adv_prob = predict_proba([adv_text])[0][1]

            orig_label = 1 if orig_prob > 0.5 else 0
            adv_label = 1 if adv_prob > 0.5 else 0

            # Only count as a flip if probability shifts significantly (> 20%)
            if orig_label != adv_label and abs(orig_prob - adv_prob) > 0.2:
                results.append("Small changes significantly impacted the prediction—potential vulnerability.")
            else:
                results.append("The model stayed consistent.")
        except Exception as e:
            print(f"Error in adversarial text: {e}")
            results.append(None)

    return results


# -----------------------------
# Unified STEF Explanations
# -----------------------------
def get_explainability_data(n_samples=20):
    """
    Generate comprehensive STEF explanations combining SHAP, attention, and adversarial testing

    Args:
        n_samples: Number of samples to process

    Returns:
        Dictionary containing test texts, labels, and STEF explanations
    """
    shap_exps = get_shap_explanations(test_texts, n_samples)
    attn_exps = [extract_attention_scores(t) for t in test_texts[:n_samples]]
    adv_exps = generate_adversarial_texts(test_texts, test_labels, n_samples)
    stef_exps = []
    for i, (text, label, shap, attn, adv) in enumerate(
            zip(test_texts[:n_samples], test_labels[:n_samples], shap_exps, attn_exps, adv_exps)):
        probs = predict_proba([text])[0]
        rumor_prob = probs[1]  # Probability for "rumor"
        # Determine predicted label and compute confidence based on the predicted class
        if rumor_prob > 0.5:
            pred_label = 1  # Model predicts "rumor"
            predicted_confidence = rumor_prob
        else:
            pred_label = 0  # Model predicts "not rumor"
            predicted_confidence = 1 - rumor_prob

        if predicted_confidence > 0.9:
            confidence_str = "very sure"
        elif predicted_confidence > 0.7:
            confidence_str = "fairly sure"
        else:
            confidence_str = "uncertain"

        stef_exp = (
            f"For: '{text}'\n"
            f"The model says it's {'a rumor' if pred_label == 1 else 'not a rumor'} and is {confidence_str} ({int(predicted_confidence * 100)}%).\n"
            f"Why: {shap or 'No clear reason.'}\n"
            f"Focus: {attn}\n"
            f"Trust: {adv or 'No issues found.'}\n"
        )
        stef_exps.append(stef_exp)
    return {
        "test_texts": test_texts[:n_samples],
        "test_labels": test_labels[:n_samples],
        "stef_explanations": stef_exps
    }


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Debug: Check label mapping for a few samples
    sample_probs = predict_proba(test_texts[:3])
    print("Sample probabilities (class 0, class 1):", sample_probs)

    data = get_explainability_data(n_samples=3)
    print("=== Simplified Transformer Explanation Framework (STEF) ===")
    for exp in data["stef_explanations"]:
        print(exp)
        print("-" * 50)

    with open("stef_results.json", "w") as f:
        json.dump(data, f, default=str)
    print("Results saved to 'stef_results.json'.")