import torch
import numpy as np
import re
import logging
import shap
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# Custom Model Wrapper for TextAttack
class CustomHuggingFaceModelWrapper(HuggingFaceModelWrapper):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def __call__(self, text_inputs):
        inputs = self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.detach().cpu().numpy()


# SHAP Masker
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


# Extract key features
def get_key_features(text, explainer):
    lpa_results = explainer.apply_linguistic_pattern_analysis(text)
    ccm_results = explainer.apply_contextual_coherence_metrics(text)
    sna_results = explainer.apply_semantic_network_analysis(text)
    key_features = set()
    for result in lpa_results:
        key_features.update(result["matches"])
    for result in ccm_results:
        pattern = result["pattern"]
        if pattern == "information gaps":
            matches = re.findall(
                r"\b(?:what if|who knows|no one knows|unclear|not sure|\?|who|what|when|where|why|how come)\b", text,
                re.IGNORECASE)
        elif pattern == "logical flow breaks":
            matches = re.findall(r"\b(?:but then again|on the other hand|even though|despite|while|whereas)\b", text,
                                 re.IGNORECASE)
        elif pattern == "temporal knowledge shifts":
            matches = re.findall(r"\b(?:knew|know|didn't know|aware|before|after|earlier|later|now)\b", text,
                                 re.IGNORECASE)
        key_features.update(matches)
    for result in sna_results:
        if result["node"] == "authority-knowledge tension":
            matches = re.findall(r"\b(?:cop|police|officer|authority|official|knew|know|didn't know|unaware|aware)\b",
                                 text, re.IGNORECASE)
        elif result["node"] == "relevance framing":
            matches = re.findall(r"\b(?:relevant|important|significant|key|critical|crucial)\b", text, re.IGNORECASE)
        elif result["node"] == "topic anchoring":
            matches = re.findall(r"\b(?:topic:|regarding:|about:|user_handle:|re:)\b", text, re.IGNORECASE)
        elif result["node"] == "nested knowledge":
            matches = re.findall(r"\b(?:know|knew)\b", text, re.IGNORECASE)
        key_features.update(matches)
    return list(key_features)


# Perturb text
def perturb_text(text, model, tokenizer, predict_proba_fn, n_perturb=5):
    model_wrapper = CustomHuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)
    perturbed_texts = []
    pred_prob = predict_proba_fn([text])[0][1].item()  # Scalar value
    for _ in range(n_perturb):
        try:
            result = attack.attack(text, 1 if pred_prob > 0.5 else 0)
            perturbed_texts.append(result.perturbed_text() if result.perturbed_text() else text)
        except Exception as e:
            logger.error(f"Perturbation failed: {e}")
            perturbed_texts.append(text)
    return perturbed_texts


# Compute metrics
def compute_stef_metrics(text, explainer, model, tokenizer, predict_proba_fn, n_perturb=5):
    tokens = tokenizer.tokenize(text)
    key_features = get_key_features(text, explainer)

    # Fidelity with SHAP
    if key_features:
        masker = SubwordMasker(tokenizer)

        def predict_fn(masks):
            masked_texts = [masker(text, mask) for mask in masks]
            probs = predict_proba_fn(masked_texts)
            return np.array([p[1] for p in probs]).reshape(-1, 1)

        # Background from perturbed texts
        background_texts = [text] + perturb_text(text, model, tokenizer, predict_proba_fn, n_perturb=2)[:2]
        background = np.ones((min(3, len(background_texts)), len(tokens)))

        try:
            explainer_shap = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer_shap.shap_values(np.ones((1, len(tokens))), nsamples=50)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values.flatten()
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            shap_values = np.zeros(len(tokens))

        # Map SHAP to key features
        feature_shap_scores = {}
        for feature in key_features:
            feature_tokens = tokenizer.tokenize(feature)
            score = sum(abs(shap_values[i]) for i, t in enumerate(tokens) if t in feature_tokens) / max(1,
                                                                                                        len(feature_tokens))
            feature_shap_scores[feature] = score

        # Mask top 20% SHAP-weighted features
        top_n = max(1, int(0.2 * len(key_features)))
        top_features = sorted(feature_shap_scores.keys(), key=lambda x: feature_shap_scores[x], reverse=True)[:top_n]
        orig_prob = predict_proba_fn([text])[0][1].item()
        masked_text = text
        for feature in top_features:
            masked_text = re.sub(r'\b' + re.escape(feature) + r'\b', "[UNK]", masked_text, flags=re.IGNORECASE)
        masked_prob = predict_proba_fn([masked_text])[0][1].item()
        fidelity = abs(orig_prob - masked_prob)
    else:
        fidelity = 0.0

    # Stability
    perturbed_texts = perturb_text(text, model, tokenizer, predict_proba_fn, n_perturb)
    perturbed_features = [set(get_key_features(pt, explainer)) for pt in perturbed_texts]
    orig_features = set(key_features)
    if orig_features:
        jaccard_scores = [len(orig_features & pf) / len(orig_features | pf) if orig_features | pf else 1.0
                          for pf in perturbed_features]
        stability = np.mean(jaccard_scores)
    else:
        stability = 1.0

    # Sparsity (fixed to match your last run)
    key_feature_tokens = set([f.lower() for f in key_features])
    token_set = set([t.lower().replace('Ġ', '') for t in tokens])
    sparsity = len(key_feature_tokens & token_set) / len(token_set) if token_set else 0.0

    return {"fidelity": fidelity, "stability": stability, "sparsity": sparsity}


# Evaluate metrics
def evaluate_stef_metrics(texts, labels, explainer, model, tokenizer, predict_proba_fn, n_samples=20):
    results = []
    for text, label in zip(texts[:n_samples], labels[:n_samples]):
        if not text.strip() or text.isspace():
            continue
        try:
            metrics = compute_stef_metrics(text, explainer, model, tokenizer, predict_proba_fn)
            logger.info(f"Text: '{text}'")
            logger.info(
                f"Metrics: Fidelity={metrics['fidelity']:.2f}, Stability={metrics['stability']:.2f}, Sparsity={metrics['sparsity']:.2f}")
            results.append({
                "text": text,
                "label": label,
                "fidelity": metrics["fidelity"],
                "stability": metrics["stability"],
                "sparsity": metrics["sparsity"]
            })
        except Exception as e:
            logger.error(f"Error computing metrics for '{text}': {e}")

    if results:
        avg_fidelity = np.mean([r["fidelity"] for r in results])
        avg_stability = np.mean([r["stability"] for r in results])
        avg_sparsity = np.mean([r["sparsity"] for r in results])
        logger.info(
            f"Average Metrics (n={len(results)}): Fidelity={avg_fidelity:.2f}, Stability={avg_stability:.2f}, Sparsity={avg_sparsity:.2f}")

    return results


if __name__ == "__main__":
    from enhanced_explainability import NovelRumorExplainer, predict_proba, test_texts, test_label
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = "./fine_tuned_roberta_augmented"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    explainer = NovelRumorExplainer()
    results = evaluate_stef_metrics(test_texts, test_label, explainer, model, tokenizer, predict_proba, n_samples=20)

    print("\n=== STEF Metrics ===")
    for r in results:
        print(f"Text: '{r['text']}'")
        print(f"Label: {'rumor' if r['label'] == 1 else 'non-rumor'}")
        print(f"Fidelity: {r['fidelity']:.2f}, Stability: {r['stability']:.2f}, Sparsity: {r['sparsity']:.2f}")
        print("-" * 50)