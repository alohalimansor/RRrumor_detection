import re
import numpy as np
import pandas as pd  # Add this

from explainblity_2 import preprocess_texts, get_shap_explanations, extract_attention_scores, \
    generate_adversarial_texts, predict_proba, RED, RESET

# Define test_texts and test_label here
file_path = "augmented_twitter_rumor_dataset.csv"  # Adjust path if needed
df = pd.read_csv(file_path, encoding="utf-8").dropna(subset=["text", "label"])
test_texts = [
    str(t) if pd.notna(t) and isinstance(t, (str, float, np.number)) else ""
    for t in df["text"].tolist()[:100]
]
test_label = df["label"].astype(int).tolist()[:100]


class NovelRumorExplainer:
    """A novel multi-level explanation for rumor detection Enhanced STEF .

    This framework analyzes text for rumor characteristics using three complementary approaches:
    1. Linguistic Pattern Analysis (LPA) - Identifies linguistic markers of rumors
    2. Contextual Coherence Metrics (CCM) - Analyzes narrative structure and coherence
    3. Semantic Network Analysis (SNA) - Identifies conceptual relationships in rumors
    """

    def __init__(self):
        # Initialize the linguistic patterns
        self.linguistic_patterns = [
            {
                "name": "hedging",
                "description": "uses uncertainty markers common in rumors",
                "patterns": [
                    (r"\b(?:possibly|maybe|perhaps|allegedly|rumor|unconfirmed)\b", 0.8),
                    (r"\b(?:might|could|would|should)\b", 0.5),
                    (r"(?:reportedly|supposedly|claim(?:s|ed)?)", 0.7)
                ]
            },
            {
                "name": "contradiction",
                "description": "contains narrative inconsistencies typical in rumors",
                "patterns": [
                    (r"\b(?:but|however|although|despite|yet|still|nevertheless)\b", 0.6),
                    (r"\b(?:didn't|wasn't|isn't|aren't|won't)\b", 0.7),
                    (r"\b(?:contrary|opposite|instead|rather)\b", 0.5)
                ]
            },
            {
                "name": "emphasis",
                "description": "uses strong emphasis which is common in rumor spreading",
                "patterns": [
                    (r"[A-Z]{2,}", 0.9),  # ALL CAPS words
                    (r"\b(?:never|always|must|obvious(?:ly)?|actual(?:ly)?)\b", 0.6),
                    (r"!+", 0.7)  # Exclamation marks
                ]
            },
            {
                "name": "sourcing",
                "description": "refers to authorities or sources in patterns typical of rumors",
                "patterns": [
                    (r"\b(?:cop|police|officer|official|government|authority)\b", 0.7),
                    (r"\b(?:expert|source|report|according|says|stated)\b", 0.6),
                    (r"\b(?:anonymous|unnamed|insider|reliable)\b", 0.8)
                ]
            },
            {
                "name": "controversy",
                "description": "mentions controversial topics that often appear in rumors",
                "patterns": [
                    (r"\b(?:shooting|killed|murder|death|violence|attack)\b", 0.8),
                    (r"\b(?:ferguson|robbery|protest|controversy|scandal)\b", 0.7),
                    (r"\b(?:conspiracy|cover(-|\s)?up|scam|hoax)\b", 0.9)
                ]
            }
        ]

    def apply_linguistic_pattern_analysis(self, text):
        """Apply linguistic pattern analysis to text."""
        results = []

        for pattern in self.linguistic_patterns:
            matches = []
            total_weight = 0

            for regex, weight in pattern["patterns"]:
                found = re.findall(regex, text, re.IGNORECASE)
                if found:
                    matches.extend(found)
                    total_weight += weight * len(found)

            if matches:
                results.append({
                    "pattern": pattern["name"],
                    "description": pattern["description"],
                    "matches": list(set(matches)),  # Remove duplicates
                    "weight": total_weight
                })

        # Sort by weight descending
        return sorted(results, key=lambda x: x["weight"], reverse=True)

    def apply_contextual_coherence_metrics(self, text):
        """Apply contextual coherence metrics to text."""
        results = []

        # Information gaps
        if re.search(r"\b(?:what if|who knows|no one knows|unclear|not sure)\b", text, re.IGNORECASE) or \
                re.search(r"(?:\?|who|what|when|where|why|how come)\b", text, re.IGNORECASE):
            results.append({
                "pattern": "information gaps",
                "description": "contains information gaps typical in rumors"
            })

        # Logical flow breaks
        if re.search(r"\b(?:but then again|on the other hand|even though|despite|while|whereas)\b", text,
                     re.IGNORECASE):
            results.append({
                "pattern": "logical flow breaks",
                "description": "shows logical inconsistencies common in rumor narratives"
            })

        # Temporal shifts with knowledge indicators
        if re.search(r"\b(?:knew|know|didn't know|aware)\b", text, re.IGNORECASE) and \
                re.search(r"\b(?:before|after|earlier|later|now)\b", text, re.IGNORECASE):
            results.append({
                "pattern": "temporal knowledge shifts",
                "description": "contains shifting knowledge states typical in rumors"
            })

        return results

    def apply_semantic_network_analysis(self, text):
        """Apply semantic network analysis to text."""
        results = []

        # Authority-knowledge tension
        if re.search(r"\b(?:cop|police|officer|authority|official)\b", text, re.IGNORECASE) and \
                re.search(r"\b(?:knew|know|didn't know|unaware|aware)\b", text, re.IGNORECASE):
            results.append({
                "node": "authority-knowledge tension",
                "description": "highlights tension between authorities and knowledge"
            })

        # Relevance framing
        if re.search(r"\b(?:relevant|important|significant|key|critical|crucial)\b", text, re.IGNORECASE):
            results.append({
                "node": "relevance framing",
                "description": "frames information as especially relevant"
            })

        # Topic anchoring
        if re.search(r"\b(?:topic:|regarding:|about:|user_handle:|re:)\b", text, re.IGNORECASE):
            results.append({
                "node": "topic anchoring",
                "description": "explicitly anchors text to a controversial topic"
            })

        # Nested knowledge structures (X knew that Y knew that Z...)
        knew_count = len(re.findall(r"\b(?:know|knew)\b", text, re.IGNORECASE))
        if knew_count >= 2:
            results.append({
                "node": "nested knowledge",
                "description": "uses complex knowledge attribution typical in rumors"
            })

        return results

    def explain(self, text, concise=True, is_rumor=True):
        """Generate a comprehensive explanation for the text.

        Args:
            text: The text to explain
            concise: If True, generate a more concise explanation
            is_rumor: Whether the model classified this as a rumor (True) or non-rumor (False)
        """
        lpa_results = self.apply_linguistic_pattern_analysis(text)
        ccm_results = self.apply_contextual_coherence_metrics(text)
        sna_results = self.apply_semantic_network_analysis(text)

        explanation = []

        # Make explanation more concise for readability
        if concise:
            # For rumor classification
            if is_rumor:
                # Combined linguistic patterns
                if lpa_results:
                    top_patterns = lpa_results[:2]  # Limit to top 2 patterns
                    pattern_items = []
                    for pattern in top_patterns:
                        matches = "', '".join(pattern["matches"][:2])  # Limit to top 2 matches
                        pattern_items.append(f"{pattern['description']} ('{matches}')")

                    if pattern_items:
                        explanation.append(f"Linguistic markers: text {' and '.join(pattern_items)}.")

                # Combined structure metrics (CCM + SNA)
                structure_items = []
                if ccm_results:
                    ccm_item = ccm_results[0]["description"] if ccm_results else None
                    if ccm_item:
                        structure_items.append(ccm_item)

                if sna_results:
                    sna_item = sna_results[0]["description"] if sna_results else None
                    if sna_item:
                        structure_items.append(sna_item)

                if structure_items:
                    explanation.append(f"Structure analysis: text {' and '.join(structure_items)}.")

            # For non-rumor classification, explain why it's NOT a rumor
            else:
                if "Official" in text or "confirmed" in text or "report" in text.lower():
                    explanation.append(
                        f"Source credibility: text uses official or verified sources which counters rumor classification.")

                # Look for specific language patterns that indicate factual reporting
                if re.search(r"\b(?:confirm|verify|according to|statement|report)\b", text, re.IGNORECASE):
                    explanation.append(
                        f"Attribution: text properly attributes information to sources rather than speculation.")

                # Look for specific details that indicate factual reporting
                if re.search(
                        r"\b(?:\d+:\d+|January|February|March|April|May|June|July|August|September|October|November|December)\b",
                        text):
                    explanation.append(
                        f"Specificity: text provides concrete details (times, dates) characteristic of factual reporting.")

                # If we didn't find any positive non-rumor indicators, explain what rumor patterns are missing
                if not explanation:
                    missing_patterns = []
                    if not any(p["name"] == "hedging" for p in lpa_results):
                        missing_patterns.append("uncertainty markers")
                    if not any(p["name"] == "contradiction" for p in lpa_results):
                        missing_patterns.append("contradictory elements")
                    if not any(p["name"] == "emphasis" for p in lpa_results):
                        missing_patterns.append("excessive emphasis")

                    if missing_patterns:
                        explanation.append(
                            f"Absent patterns: text lacks {', '.join(missing_patterns)} typically found in rumors.")
                    else:
                        explanation.append(
                            f"Balance assessment: despite some rumor-like patterns, the overall language structure is more consistent with factual reporting.")

        # Original more detailed explanation
        else:
            # Add explanation based on classification
            rumor_status = "supporting a rumor classification" if is_rumor else "which would normally suggest a rumor, but other factors outweigh these"

            # Add LPA results (linguistic patterns)
            if lpa_results:
                top_patterns = lpa_results[:3]  # Top 3 patterns
                for pattern in top_patterns:
                    matches = "', '".join(pattern["matches"])
                    explanation.append(f"This text {pattern['description']} ('{matches}'), {rumor_status}.")

            # Add CCM results (coherence metrics)
            if ccm_results:
                for result in ccm_results:
                    explanation.append(f"It {result['description']}, {rumor_status}.")

            # Add SNA results (semantic networks)
            if sna_results:
                for result in sna_results:
                    explanation.append(f"The narrative {result['description']}, {rumor_status}.")

        # If we couldn't find any patterns
        if not explanation:
            if is_rumor:
                return "No strong linguistic patterns detected—model likely using subtle content indicators for rumor classification."
            else:
                return "Text shows characteristics of factual reporting—lacks typical rumor linguistic patterns."

        return "\n- ".join([""] + explanation)


def get_novel_rumor_explanation(text):
    """Standalone function to generate novel explanations for rumor text."""
    explainer = NovelRumorExplainer()
    return explainer.explain(text)


def get_enhanced_explainability_data(selected_texts, selected_label):
    """Drop-in replacement for get_explainability_data that uses the novel explainer.

    This function maintains full compatibility with the original interface
    but enhances the explanation quality with the multi-level explainer.
    """
    # Keep the global assignment as in the original code
    global test_texts, test_label
    test_texts = selected_texts
    test_label = selected_label
    n_samples = len(selected_texts)

    # Initialize the novel explainer
    novel_explainer = NovelRumorExplainer()

    # Use the original functions for compatibility
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

        # Generate the novel explanation with the model's classification passed in
        is_rumor = pred_label == 1
        novel_exp = novel_explainer.explain(text, concise=True, is_rumor=is_rumor)

        # Fall back to SHAP if novel explainer doesn't find patterns
        final_exp = novel_exp
        if "No strong linguistic patterns detected" in novel_exp:
            final_exp = shap_exp or "No explanation available."

        stef_exp = (
            f"For: '{text}'\n"
            f"True label: {'rumor' if label == 1 else 'non-rumor'}\n"
            f"The model says it's {'a rumor' if pred_label == 1 else 'not a rumor'} and is {confidence_str} ({int(predicted_confidence * 100)}%).\n"
            f"Why:{final_exp}\n"
            f"Focus: {attn}\n"
            f"Trust: {RED + adv + RESET if adv and 'flipped' in adv else adv or 'No issues found.'}\n"
        )
        stef_exps.append(stef_exp)

    return {
        "test_texts": test_texts[:n_samples],
        "test_label": test_label[:n_samples],
        "stef_explanations": stef_exps
    }


def demonstrate_rumor_nonrumor_examples():
    """Generate explicit examples of rumor and non-rumor explanations."""
    print("\n\n===== RUMOR & NON-RUMOR EXPLANATION DEMONSTRATION =====\n")

    # Define our examples
    rumor_text = "but mike brown knew about the robbery and DIDN'T know the cop didn't know - that's what makes it POSSIBLY relevant user_handle: DefinitelyMay_b topic: ferguson"

    non_rumor_text = "Official statement: Police confirmed that the incident took place at 2:30 PM. Two witnesses have provided testimony. Investigation is ongoing. user_handle: NewsUpdates topic: downtown"

    # Initialize the novel explainer
    novel_explainer = NovelRumorExplainer()

    # RUMOR EXAMPLE
    print("===== RUMOR EXAMPLE =====")
    print(f"Text: '{rumor_text}'")

    # Get prediction (should be rumor)
    probs = predict_proba([rumor_text])[0]
    rumor_prob = probs[1]
    is_rumor = rumor_prob > 0.5
    confidence = rumor_prob if is_rumor else 1 - rumor_prob

    print(f"Model prediction: {'Rumor' if is_rumor else 'Non-rumor'} with {int(confidence * 100)}% confidence")

    # Get explanation
    explanation = novel_explainer.explain(rumor_text, concise=True, is_rumor=is_rumor)
    print(f"Why:{explanation}")

    # Add these lines to include Focus and Trust
    attn = extract_attention_scores(rumor_text)
    adv = generate_adversarial_texts([rumor_text], [1], 1)[0]
    print(f"Focus: {attn}")
    print(f"Trust: {RED + adv + RESET if adv and 'flipped' in adv else adv or 'No issues found.'}")

    # NON-RUMOR EXAMPLE
    print("\n===== NON-RUMOR EXAMPLE =====")
    print(f"Text: '{non_rumor_text}'")

    # Get prediction (should be non-rumor)
    probs = predict_proba([non_rumor_text])[0]
    rumor_prob = probs[1]
    is_rumor = rumor_prob > 0.5
    confidence = rumor_prob if is_rumor else 1 - rumor_prob

    print(f"Model prediction: {'Rumor' if is_rumor else 'Non-rumor'} with {int(confidence * 100)}% confidence")

    # Get explanation
    explanation = novel_explainer.explain(non_rumor_text, concise=True, is_rumor=is_rumor)
    print(f"Why:{explanation}")

    # Add these lines to include Focus and Trust
    attn = extract_attention_scores(non_rumor_text)
    adv = generate_adversarial_texts([non_rumor_text], [0], 1)[0]
    print(f"Focus: {attn}")
    print(f"Trust: {RED + adv + RESET if adv and 'flipped' in adv else adv or 'No issues found.'}")

    print("\n===== END OF DEMONSTRATION =====")


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
        test_label_np = np.append(test_label_np, 1)  # Assuming it's a rumor
        selected_indices.append(len(test_texts_np) - 1)

    selected_texts = test_texts_np[selected_indices].tolist()
    selected_label = test_label_np[selected_indices].tolist()
    demonstrate_rumor_nonrumor_examples()
    # Predict probabilities for selected samples
    sample_probs = predict_proba(selected_texts)
    print("Sample probabilities (class 0, class 1):")
    for i, (text, prob, true_label) in enumerate(zip(selected_texts, sample_probs, selected_label)):
        print(f"Text {i + 1} (True label: {'rumor' if true_label == 1 else 'non-rumor'}): {prob}")

    # Generate Simplified STEF explanations
    data = get_enhanced_explainability_data(selected_texts, selected_label)
    print("\n=== Simplified Transformer Explanation Framework (STEF) ===")
    for exp in data["stef_explanations"]:
        print(exp)
        print("-" * 50)