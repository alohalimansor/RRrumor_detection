# STEF: A Robust and Explainable Transformer Framework for Rumor Detection

This repository contains the implementation of the Simplified Transformer Explanation Framework (STEF), an advanced approach to rumor detection that combines transformer-based classification with multi-level explainability mechanisms.

## Overview

STEF integrates a fine-tuned adversarial RoBERTa model with three complementary explainability components to provide human-interpretable insights into rumor detection:

1. **Linguistic Pattern Analysis (LPA)**: Identifies linguistic markers of rumors (hedging, contradiction, emphasis, sourcing, controversy)
2. **Contextual Coherence Metrics (CCM)**: Analyzes narrative structure and coherence (information gaps, logical flow breaks, temporal knowledge shifts)
3. **Semantic Network Analysis (SNA)**: Maps conceptual relationships (authority-knowledge tension, relevance framing, topic anchoring, nested knowledge structures)

## Key Features

- **State-of-the-art performance**: 95.84% accuracy and 95.81% F1-score on rumor detection
- **Adversarial robustness**: Reduces classification flip rate from 18% to 8% under adversarial attacks
- **Multi-level explainability**: Provides human-interpretable insights beyond token-level attributions
- **Enhanced evaluation**: Assesses explanations using fidelity, stability, and sparsity metrics
- **Class-balanced training**: Uses Focal Loss with weighted cross-entropy to address class imbalance
- **Multi-model comparison**: Benchmarks against BERT, RoBERTa, DeBERTa, XLNet, T5, and MisRoBÆRTa models

## Directory Structure

```
.
├── README.md
├── advanced_explain_roberta.py      # Advanced SHAP explainability for RoBERTa
├── augment_rumor_dataset.py         # Adversarial data augmentation
├── clean_rumor_dataset.py           # Dataset preprocessing
├── comparison.py                    # Comparison with baseline methods
├── computations.py                  # Evaluation metrics implementation
├── enhanced_explainability.py       # STEF framework implementation
├── evaluate_baseline_model.py       # Baseline model evaluation
├── evaluate_roberta.py              # RoBERTa evaluation
├── evaluate_roberta_augmented.py    # Adversarial RoBERTa evaluation
├── evaluate_t5.py                   # T5 model evaluation
├── explainability.py                # Core explainability functions
├── explainblity_2.py                # Extended explainability analyses
├── fine_tune_roberta.py             # Fine-tuning script for RoBERTa
├── fine_tune_roberta_augmented.py   # Fine-tuning with adversarial examples
├── misinformation_model_testing.py  # Comprehensive model evaluation script
├── MisRo_Train.py                   # Training script for MisRoBÆRTa model
├── MisRob_eval.py                   # Evaluation script for MisRoBÆRTa
├── pretrain_roberta_mlm.py          # Self-supervised pre-training with MLM
├── Seprate_Explain.py               # Ablation study of STEF components
├── SNA_Graph.py                     # Graph generation for Semantic Network Analysis
├── stef_metrics.py                  # STEF metrics calculation
└── train_hybrid_transformer.py      # Hybrid model with BERT, RoBERTa and DeBERTa
```

## Installation and Requirements

```bash
# Clone the repository
git clone https://github.com/alohalimansor/RRrumor_detection.git
cd RRrumor_detection

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 1.9.0
- Transformers >= 4.24.0
- SHAP >= 0.41.0
- TextAttack >= 0.3.7
- Scikit-learn >= 1.0.2
- Pandas >= 1.4.0
- NumPy >= 1.22.0
- Matplotlib >= 3.5.0

## Usage

### Preprocessing and Data Augmentation

```bash
# Clean and preprocess the dataset
python clean_rumor_dataset.py

# Apply adversarial data augmentation
python augment_rumor_dataset.py
```

### Model Training

```bash
# Pre-train RoBERTa with Masked Language Modeling
python pretrain_roberta_mlm.py

# Fine-tune RoBERTa on the rumor detection task
python fine_tune_roberta.py

# Fine-tune with adversarial examples and class balancing
python fine_tune_roberta_augmented.py

# Train the hybrid ensemble of BERT, RoBERTa, and DeBERTa
python train_hybrid_transformer.py

# Train the MisRoBÆRTa model (specialized for misinformation)
python MisRo_Train.py
```

### Evaluation and Testing

```bash
# Evaluate the baseline model
python evaluate_baseline_model.py

# Evaluate the fine-tuned RoBERTa model
python evaluate_roberta.py

# Evaluate the adversarially trained RoBERTa model
python evaluate_roberta_augmented.py

# Evaluate the T5 model
python evaluate_t5.py

# Evaluate MisRoBÆRTa model performance
python MisRob_eval.py

# Comprehensive testing across all models
python misinformation_model_testing.py
```

### Explainability Analysis

```bash
# Generate STEF explanations
python enhanced_explainability.py

# Compare explainability methods
python comparison.py

# Run the ablation study on STEF components
python Seprate_Explain.py

# Generate Semantic Network Analysis visualization
python SNA_Graph.py

# Compute detailed STEF metrics
python stef_metrics.py
```

## Examples

### Sample STEF Explanation

```
Text: 'but mike brown knew about the robbery and DIDN'T know the cop didn't know - that's what makes it POSSIBLY relevant'
The model says it's a rumor and is very sure (95%).
Why: Linguistic markers: text uses uncertainty markers common in rumors ('POSSIBLY') and contains contradictory elements typical in rumors ('but', 'DIDN'T').
Structure analysis: text contains shifting knowledge states typical in rumors and uses complex knowledge attribution typical in rumors.
Focus: The model focused on: DIDN'T, POSSIBLY, robbery.
Trust: The model stayed consistent.
```

### Comparison with Other Methods

![Explanation Stability Comparison](stability_comparison.png)

## Citation

If you use this code or find our work useful for your research, please cite our paper:

```bibtex
@article{alohali2024stef,
  title={{STEF: A Robust and Explainable Transformer Framework for Rumor Detection}},
  author={Alohali, M. and [Other Authors]},
  journal={[Journal Name]},
  year={2024}
}
```

## License
Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)

By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial 4.0 International Public License ("Public License").

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

To view a copy of this license, visit:
https://creativecommons.org/licenses/by-nc/4.0/

---

© 2024 Mansour Alohali. All rights reserved.
If you use this project in your research or applications, please cite the associated paper and link to the GitHub repository.



## Contact

Mansor Alohali, Applied College, Imam Mohammad Ibn Saud Islamic University (IMSIU), Riyadh, Saudi Arabia
 malohali@imamu.edu.sa
