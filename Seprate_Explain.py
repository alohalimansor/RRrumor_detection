import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel
import nltk
from nltk.tokenize import word_tokenize
import logging
import random
from transformers import AutoTokenizer, AutoModel  # For real embeddings

nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder imports
from enhanced_explainability import NovelRumorExplainer, predict_proba, preprocess_texts, extract_attention_scores

# Load dataset
file_path = "augmented_twitter_rumor_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])
test_texts = df["text"].tolist()[:500]
test_labels = df["label"].astype(int).tolist()[:500]

# Embedding function (toggle between random and RoBERTa)
USE_ROBERTA = False  # Set to True after installing transformers
if USE_ROBERTA:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")


    def get_embedding(text):
        if not isinstance(text, str):
            return np.zeros(768)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
else:
    def get_embedding(text):
        if not isinstance(text, str):
            return np.zeros(50)
        tokens = word_tokenize(text.lower())
        return np.mean([np.random.rand(50) for _ in tokens], axis=0) if tokens else np.zeros(50)


# Helper function to convert explanation items to strings
def stringify_explanation_item(item):
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        return " ".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float)))
    elif isinstance(item, (list, tuple)):
        return " ".join(stringify_explanation_item(x) for x in item)
    else:
        return str(item)


# Stronger dummy adversarial text generator
def generate_adversarial_texts(texts, labels, num_examples):
    synonyms = {
        "good": "great", "bad": "terrible", "said": "claimed", "is": "seems",
        "people": "folks", "big": "huge", "small": "tiny", "know": "understand",
        "might": "definitely", "maybe": "certainly"
    }
    perturbed = []
    for text in texts:
        words = text.split()
        perturbed_words = []
        for word in words:
            if word.lower() in synonyms and random.random() > 0.25:
                perturbed_words.append(synonyms[word.lower()])
            elif random.random() > 0.9:
                continue
            else:
                perturbed_words.append(word)
        if random.random() > 0.9:
            perturbed_words.append(random.choice(list(synonyms.values())))
        perturbed.append(" ".join(perturbed_words))
    return perturbed[:num_examples]


# Improved STEFVariant class
class STEFVariant:
    def __init__(self, remove_lpa=False, remove_ccm=False, remove_sna=False):
        self.remove_lpa = remove_lpa
        self.remove_ccm = remove_ccm
        self.remove_sna = remove_sna
        self.explainer = NovelRumorExplainer()

    def explain(self, text, is_rumor):
        if not isinstance(text, str):
            return ["Error: Invalid input text"]

        explanation = []
        if not self.remove_lpa:
            lpa_result = self.explainer.apply_linguistic_pattern_analysis(text)
            if lpa_result:
                explanation.extend([stringify_explanation_item(r) for r in
                                    (lpa_result if isinstance(lpa_result, list) else [lpa_result])])
            else:
                explanation.append("LPA: No markers detected")
        if not self.remove_ccm:
            ccm_result = self.explainer.apply_contextual_coherence_metrics(text)
            if ccm_result:
                explanation.extend([stringify_explanation_item(r) for r in
                                    (ccm_result if isinstance(ccm_result, list) else [ccm_result])])
            else:
                explanation.append("CCM: No coherence issues")
        if not self.remove_sna:
            sna_result = self.explainer.apply_semantic_network_analysis(text)
            if sna_result:
                explanation.extend([stringify_explanation_item(r) for r in
                                    (sna_result if isinstance(sna_result, list) else [sna_result])])
            else:
                explanation.append("SNA: No relationships mapped")
        return explanation if explanation else ["Default: Minimal explanation"]


# Improved evaluation function with label-split stability
def evaluate_model(model, texts, labels):
    fidelity_scores, stability_scores, sparsity_scores = [], [], []
    rumor_stability, nonrumor_stability = [], []
    for i, text in enumerate(texts):
        is_rumor = labels[i] == 1
        original_explanation = model.explain(text, is_rumor)

        try:
            adv_texts = generate_adversarial_texts([text], [labels[i]], 1)
            adv_text = adv_texts[0] if adv_texts and adv_texts[0] is not None else text
        except Exception as e:
            logging.error(f"Error in adversarial text for '{text}': {str(e)}")
            adv_text = text

        adv_explanation = model.explain(adv_text, is_rumor)

        orig_text = " ".join(original_explanation)
        adv_text_str = " ".join(adv_explanation)
        orig_emb = get_embedding(orig_text)
        adv_emb = get_embedding(adv_text_str)
        fidelity = cosine_similarity([orig_emb], [adv_emb])[0][0]
        fidelity_scores.append(fidelity)

        stability = 1 if original_explanation == adv_explanation else 0
        stability_scores.append(stability)
        if is_rumor:
            rumor_stability.append(stability)
        else:
            nonrumor_stability.append(stability)

        sparsity = len(orig_text.split()) / max(1, len(text.split()))
        sparsity_scores.append(sparsity)

    # Debug: Log explanations for first few samples
    for i in range(min(3, len(texts))):
        logging.info(
            f"{model.__class__.__name__} Sample {i} (Label {labels[i]}): {model.explain(texts[i], labels[i] == 1)}")

    return (np.mean(fidelity_scores), np.mean(stability_scores), np.mean(sparsity_scores),
            fidelity_scores, stability_scores, sparsity_scores,
            np.mean(rumor_stability) if rumor_stability else 0,
            np.mean(nonrumor_stability) if nonrumor_stability else 0)


# Define the four models
full_stef = STEFVariant()
no_lpa = STEFVariant(remove_lpa=True)
no_ccm = STEFVariant(remove_ccm=True)
no_sna = STEFVariant(remove_sna=True)

# Filter for rumor-only data
rumor_texts = [t for t, l in zip(test_texts, test_labels) if l == 1]
rumor_labels = [l for l in test_labels if l == 1]
print(f"Rumor-only dataset size: {len(rumor_texts)} samples")

# Run evaluations on rumor-only data
models = {"Full STEF": full_stef, "No LPA": no_lpa, "No CCM": no_ccm, "No SNA": no_sna}
results = {}
raw_scores = {}
for name, model in models.items():
    fidelity, stability, sparsity, fid_scores, stab_scores, spar_scores, rumor_stab, nonrumor_stab = evaluate_model(
        model, rumor_texts, rumor_labels)
    results[name] = {"Fidelity": fidelity, "Stability": stability, "Sparsity": sparsity,
                     "Rumor Stability": rumor_stab, "Non-rumor Stability": nonrumor_stab}
    raw_scores[name] = {"Fidelity": fid_scores, "Stability": stab_scores, "Sparsity": spar_scores}

# Display results
results_df = pd.DataFrame(results).T
print("Mean Results (Rumor-Only):")
print(results_df)

# Statistical analysis
print("\nStatistical Significance (p-values from paired t-tests vs Full STEF, Rumor-Only):")
for metric in ["Fidelity", "Stability", "Sparsity"]:
    print(f"\n{metric}:")
    for name in ["No LPA", "No CCM", "No SNA"]:
        t_stat, p_val = ttest_rel(raw_scores["Full STEF"][metric], raw_scores[name][metric])
        print(f"{name}: p = {p_val:.4f} {'(significant)' if p_val < 0.05 else ''}")