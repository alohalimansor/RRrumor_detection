import torch
import shap
import numpy as np
import pandas as pd
import os
import logging
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ANSI color codes for console output
RED = "\033[91m"
RESET = "\033[0m"

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Model and Tokenizer Setup
model_path = "./fine_tuned_roberta_augmented"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path '{model_path}' not found. Please train or download the model.")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, attn_implementation="eager")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Data Loading
file_path = "augmented_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])
test_texts = [
    str(t) if pd.notna(t) and isinstance(t, (str, float, np.number)) else ""
    for t in df["text"].tolist()[:100]
]
test_label = df["label"].astype(int).tolist()[:100]
background_texts = test_texts[:50]

# Check label distribution
print(
    f"Label distribution in test_label[:100]: Rumors (1): {sum(test_label)}, Non-rumors (0): {len(test_label) - sum(test_label)}")


# Preprocessing Function
def preprocess_texts(texts, batch_size=16):
    """Tokenize a list of texts in batches with consistent padding."""
    all_inputs = {"input_ids": [], "attention_mask": []}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        logger.info(f"Batch {i // batch_size}: input_ids shape = {inputs['input_ids'].shape}")
        all_inputs["input_ids"].append(inputs["input_ids"])
        all_inputs["attention_mask"].append(inputs["attention_mask"])
    try:
        return {
            "input_ids": torch.cat(all_inputs["input_ids"]).to(device),
            "attention_mask": torch.cat(all_inputs["attention_mask"]).to(device)
        }
    except RuntimeError as e:
        logger.error(f"Error in concatenation: {e}")
        raise


# Prediction Function
def predict_proba(texts, batch_size=16):
    """Predict rumor probabilities for a list of texts."""
    if isinstance(texts, str):
        texts = [texts]
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        probs.extend(batch_probs.tolist())
    return np.array(probs)


# SHAP Explanations
class SubwordMasker:
    def __init__(self, tokenizer, mask_token="[UNK]"):
        self.tokenizer = tokenizer
        self.mask_token = mask_token

    def __call__(self, text, mask):
        tokens = self.tokenizer.tokenize(text)
        if len(mask) > len(tokens):
            mask = mask[:len(tokens)]
        masked_tokens = [t if mask[i] else self.mask_token for i, t in enumerate(tokens)]
        return self.tokenizer.convert_tokens_to_string(masked_tokens)


def compute_metrics(text, shap_values, tokens, n_perturb=5):
    """Compute fidelity, stability, and sparsity metrics for SHAP explanations."""
    masker = SubwordMasker(tokenizer)
    top_indices = np.argsort(np.abs(shap_values))[-int(0.2 * len(tokens)):]
    mask = [0 if i in top_indices else 1 for i in range(len(tokens))]
    masked_text = masker(text, mask)
    orig_prob = predict_proba([text])[0][1]
    masked_prob = predict_proba([masked_text])[0][1]
    fidelity = abs(orig_prob - masked_prob)

    perturbed_texts = [masker(text, np.random.randint(0, 2, len(tokens))) for _ in range(n_perturb)]
    perturbed_probs = predict_proba(perturbed_texts)
    stability = np.mean([1 if (p[1] > 0.5) == (orig_prob > 0.5) else 0 for p in perturbed_probs])

    threshold = 0.0001  # Lowered threshold
    sparsity = np.mean(np.abs(shap_values) > threshold)

    return {"fidelity": fidelity, "stability": stability, "sparsity": sparsity}


def get_shap_explanations(texts, n_samples=20):
    """Generate SHAP explanations for a sample of texts."""
    explanations = []
    masker = SubwordMasker(tokenizer)
    background_inputs = preprocess_texts(background_texts)

    for text in texts[:n_samples]:
        if not isinstance(text, str) or not text.strip() or text.isspace():
            explanations.append(None)
            continue
        try:
            tokens = tokenizer.tokenize(text)

            def predict_fn(masks):
                masked_texts = [masker(text, mask) for mask in masks]
                probs = predict_proba(masked_texts)
                return np.array([p[1] for p in probs]).reshape(-1, 1)

            background = np.ones((10, len(tokens)))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(np.ones((1, len(tokens))))

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            logger.info(f"SHAP values for '{text}': {list(zip(tokens, shap_values.flatten()))}")
            summary = categorize_and_summarize(tokens, shap_values.flatten())
            metrics = compute_metrics(text, shap_values.flatten(), tokens)
            logger.info(
                f"Metrics for '{text}': Fidelity={metrics['fidelity']:.2f}, "
                f"Stability={metrics['stability']:.2f}, Sparsity={metrics['sparsity']:.2f}")
            explanations.append(summary)
        except Exception as e:
            logger.error(f"Error in SHAP for '{text}': {e}")
            explanations.append(None)
    return explanations


def categorize_and_summarize(words, shap_values, threshold=0.0001):
    """Categorize and summarize SHAP contributions with enhanced rumor patterns."""
    # Define categories that better capture rumor characteristics
    uncertainty_words = set(
        ["possibly", "maybe", "rumor", "alleged", "supposedly", "might", "could", "perhaps", "unconfirmed"])
    contradiction_words = set(["didn't", "but", "however", "although", "despite", "nevertheless", "yet", "contrary"])
    authority_words = set(["cop", "police", "officer", "official", "government", "authority", "expert", "source"])
    controversy_words = set(["robbery", "ferguson", "shooting", "killed", "violence", "protest", "controversial"])
    emphasis_words = set(["must", "never", "all", "none", "always", "actually", "obviously"])

    summary = {
        "uncertainty": [],
        "contradiction": [],
        "authority": [],
        "controversy": [],
        "emphasis": [],
        "other": []
    }

    # Check for uppercase words (potential emphasis)
    uppercase_words = set()
    for word in words:
        clean_word = word.replace('Ġ', '')
        if any(c.isupper() for c in clean_word) and len(clean_word) > 1:
            uppercase_words.add(word)

    for word, value in zip(words, shap_values):
        if abs(value) < threshold:
            continue

        clean_word = word.lower().replace('Ġ', '')

        # Check for uppercase emphasis
        if word in uppercase_words:
            summary["emphasis"].append((word, value))
            continue

        # Check categories
        if clean_word in uncertainty_words:
            summary["uncertainty"].append((word, value))
        elif clean_word in contradiction_words:
            summary["contradiction"].append((word, value))
        elif clean_word in authority_words:
            summary["authority"].append((word, value))
        elif clean_word in controversy_words:
            summary["controversy"].append((word, value))
        elif clean_word in emphasis_words:
            summary["emphasis"].append((word, value))
        else:
            summary["other"].append((word, value))

    return format_improved_summary(summary)


def format_improved_summary(summary):
    """Format the summary with meaningful explanations."""
    explanations = {
        "uncertainty": "uses uncertain language suggesting speculation rather than fact",
        "contradiction": "contains contradictory elements that often appear in rumor narratives",
        "authority": "references authorities in a way typical of rumor contexts",
        "controversy": "mentions controversial topics or events common in rumors",
        "emphasis": "uses emphatic language or capitalization typical of rumor spreading"
    }

    text_parts = []
    for category, items in summary.items():
        if items and category != "other":
            sorted_items = sorted(items, key=lambda x: abs(x[1]), reverse=True)
            word_list = ', '.join([f"'{w.replace('Ġ', '')}'" for w, _ in sorted_items[:3]])
            direction = "suggesting rumor" if sorted_items[0][1] > 0 else "countering rumor impression"
            text_parts.append(f"The text {explanations[category]} ({word_list}) {direction}.")

    if summary["other"]:
        sorted_other = sorted(summary["other"], key=lambda x: abs(x[1]), reverse=True)
        if len(sorted_other) > 2:
            words = ', '.join([f"'{w.replace('Ġ', '')}'" for w, _ in sorted_other[:3]])
            text_parts.append(f"Other significant terms ({words}) contribute to the classification.")

    if not text_parts:
        return "No strong clues found—prediction based on overall tone."

    # Limit to top 3 explanations for clarity if we have too many
    if len(text_parts) > 3:
        text_parts = text_parts[:3]
        text_parts.append("Additional linguistic patterns also support this classification.")

    return " ".join(text_parts)


# Attention Extraction
def extract_attention_scores(text):
    """Extract top attention scores from the model’s last layer."""
    if not isinstance(text, str) or not text.strip() or text.isspace():
        return "No focus identified."
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions[-1].squeeze(0).max(dim=0).values.mean(dim=0).cpu().numpy()
    token_ids = inputs["input_ids"].squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    stopwords = set(
        ["the", "a", "an", "is", "are", "was", "and", "or", "in", "on", "at", "to", ".", ",", ":", "’", "?", "<s>",
         "</s>", "<pad>"])
    valid_pairs = [(token.replace('Ġ', ''), score) for token, score in zip(tokens, attentions) if
                   token.lower() not in stopwords and len(token) > 1]
    if not valid_pairs:
        return "No clear focus."
    top_words = sorted(valid_pairs, key=lambda x: x[1], reverse=True)[:3]
    full_text = ' '.join(word for word, _ in top_words).strip()
    return f"The model focused on: {full_text}."


# Custom Model Wrapper for TextAttack
class CustomHuggingFaceModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def __call__(self, text_inputs):
        outputs = self.model(
            **self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True).to(device))
        return outputs.logits.detach().cpu().numpy()


# Adversarial Trust Test
def generate_adversarial_texts(texts, label, n_samples=20):
    # Wrap model for TextAttack compatibility
    model_wrapper = CustomHuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)
    results = []
    for text, label in zip(texts[:n_samples], label[:n_samples]):
        if not isinstance(text, str) or not text.strip() or text.isspace():
            results.append(None)
            continue
        try:
            result = attack.attack(text, label)
            adv_text = result.perturbed_text() if result.perturbed_text() else text
            orig_prob = predict_proba([text])[0][1]
            adv_prob = predict_proba([adv_text])[0][1]
            orig_label = 1 if orig_prob > 0.5 else 0
            adv_label = 1 if adv_prob > 0.5 else 0
            shift = abs(orig_prob - adv_prob)
            if orig_label != adv_label:
                results.append(f"Prediction flipped with {shift:.2f} probability shift—potential vulnerability.")
            else:
                results.append(f"Prediction consistent; shift of {shift:.2f}.")
        except Exception as e:
            logger.error(f"Error in adversarial text for '{text}': {e}")
            results.append(None)
    return results


# Unified STEF Explanations
def get_explainability_data(selected_texts, selected_label):
    """Generate STEF explanations including prediction, focus, and trust for selected samples."""
    global test_texts, test_label
    test_texts = selected_texts
    test_label = selected_label
    n_samples = len(selected_texts)

    inputs = preprocess_texts(test_texts[:n_samples])
    shap_exps = get_shap_explanations(test_texts, n_samples)
    attn_exps = [extract_attention_scores(t) for t in test_texts[:n_samples]]
    adv_exps = generate_adversarial_texts(test_texts, test_label, n_samples)
    stef_exps = []

    for i, (text, label, shap_exp, attn, adv) in enumerate(
            zip(test_texts[:n_samples], test_label[:n_samples], shap_exps, attn_exps, adv_exps)):
        probs = predict_proba([text])[0]
        rumor_prob = probs[1]
        pred_label = 1 if rumor_prob > 0.5 else 0
        predicted_confidence = rumor_prob if pred_label == 1 else 1 - rumor_prob
        confidence_str = "very sure" if predicted_confidence > 0.9 else "fairly sure" if predicted_confidence > 0.7 else "uncertain"

        stef_exp = (
            f"For: '{text}'\n"
            f"True label: {'rumor' if label == 1 else 'non-rumor'}\n"
            f"The model says it’s {'a rumor' if pred_label == 1 else 'not a rumor'} and is {confidence_str} ({int(predicted_confidence * 100)}%).\n"
            f"Why: {shap_exp or 'No explanation available.'}\n"
            f"Focus: {attn}\n"
            f"Trust: {RED + adv + RESET if adv and 'flipped' in adv else adv or 'No issues found.'}\n"
        )
        stef_exps.append(stef_exp)

    return {
        "test_texts": test_texts[:n_samples],
        "test_label": test_label[:n_samples],
        "stef_explanations": stef_exps
    }


if __name__ == "__main__":
    # Select samples including your manually fixed rumor example
    test_texts_np = np.array(test_texts)
    test_label_np = np.array(test_label)

    # Find rumor indices and include your example
    rumor_indices = np.where(test_label_np == 1)[0]
    selected_indices = rumor_indices[:min(3, len(rumor_indices))].tolist()

    your_example = 'but mike brown knew about the robbery and DIDN\'T know the cop didn\'t know - that\'s what makes it POSSIBLY relevant user_handle: DefinitelyMay_b topic: ferguson'
    if your_example in test_texts_np:
        example_idx = np.where(test_texts_np == your_example)[0][0]
        if example_idx not in selected_indices:
            selected_indices.append(example_idx)
    else:
        test_texts_np = np.append(test_texts_np, your_example)
        test_label_np = np.append(test_label_np, 1)  # Assuming it’s a rumor
        selected_indices.append(len(test_texts_np) - 1)

    selected_texts = test_texts_np[selected_indices].tolist()
    selected_label = test_label_np[selected_indices].tolist()

    # Predict probabilities for selected samples
    sample_probs = predict_proba(selected_texts)
    print("Sample probabilities (class 0, class 1):")
    for i, (text, prob, true_label) in enumerate(zip(selected_texts, sample_probs, selected_label)):
        print(f"Text {i + 1} (True label: {'rumor' if true_label == 1 else 'non-rumor'}): {prob}")

    # Generate STEF explanations for selected samples
    data = get_explainability_data(selected_texts, selected_label)
    print("\n=== Simplified Transformer Explanation Framework (STEF) ===")
    for exp in data["stef_explanations"]:
        print(exp)
        print("-" * 50)
