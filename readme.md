# Twitter Rumor Detection with Adversarial Training and Explainable AI

This repository contains the code and resources for the paper *"Enhancing Twitter Rumor Detection with Adversarial Training and Explainable AI"* submitted to a high-ranking journal. The project implements an adversarially trained RoBERTa model with a multi-level explainability framework (Simplified Transformer Explanation Framework, STEF) to detect rumors on Twitter (X), achieving 95.84% accuracy on the PHEME dataset while improving transparency and robustness.

## Overview

The rapid spread of misinformation on social media poses significant challenges to trust and stability. This project enhances rumor detection by integrating:
- **Adversarial Training**: Using Fast Gradient Method (FGM) and TextFooler to improve model resilience.
- **Explainable AI (XAI)**: An enhanced STEF framework with Linguistic Pattern Analysis (LPA), Contextual Coherence Metrics (CCM), and Semantic Network Analysis (SNA) for interpretable predictions.
- **Quantitative Evaluation**: Metrics like fidelity, stability, and sparsity to assess explanation quality.
- **Visualizations**: Tools to illustrate model decisions via SHAP, LIME, attention heatmaps, and adversarial comparisons.

Key results include a 95.84% classification accuracy, an 8% adversarial flip rate (down from 18%), and a 21% improvement in explanation stability over SHAP baselines.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/twitter-rumor-detection.git
   cd twitter-rumor-detection
   
Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Usage
Dataset Preparation
Input Data: Place combined_twitter_rumor_dataset.csv and cleaned_rumors.csv in the data/ directory. These files are assumed to contain tweet texts and labels (0 for non-rumor, 1 for rumor).
Augmentation: Run data_augmentation.py to generate augmented_twitter_rumor_dataset.csv:
python src/data_augmentation.py

Pretraining
Pretrain RoBERTa on cleaned rumor data for domain adaptation:
python src/pretrain_mlm.py


twitter-rumor-detection/
├── data/                          # Dataset files (not included due to size; see notes)
│   ├── combined_twitter_rumor_dataset.csv
│   ├── cleaned_rumors.csv
│   └── augmented_twitter_rumor_dataset.csv
├── src/                           # Source code
│   ├── data_augmentation.py       # Adversarial data augmentation
│   ├── pretrain_mlm.py            # Self-supervised MLM pretraining
│   ├── train_hybrid_model.py      # Baseline hybrid model (BERT+RoBERTa+DeBERTa)
│   ├── train_augmented_model.py   # Final adversarially trained RoBERTa model
│   ├── enhanced_explainability.py # Enhanced STEF framework
│   ├── evaluate_stef_metrics.py   # Quantitative metric evaluation
│   └── visualize_explanations.py  # Visualization generation
├── visualization_output/          # Generated visualizations (see .gitignore)
├── requirements.txt              # Python dependencies
└── README.md                     # This file


Key Files
data_augmentation.py: Generates adversarial examples using TextAttack to enhance the dataset.
pretrain_mlm.py: Pretrains RoBERTa on cleaned rumor data for domain adaptation.
train_hybrid_model.py: Implements the hybrid baseline model from Table 2 (85.02% accuracy).
train_augmented_model.py: Trains the final RoBERTa model with FGM and Focal Loss on augmented data.
enhanced_explainability.py: Implements the enhanced STEF framework with LPA, CCM, and SNA.
evaluate_stef_metrics.py: Computes fidelity, stability, and sparsity metrics for STEF.

Notes on Data
The PHEME dataset was used in the paper but is not included here due to size and licensing. Replace combined_twitter_rumor_dataset.csv with your own labeled dataset (columns: text, label) and cleaned_rumors.csv with preprocessed text data for pretraining.
Run data_augmentation.py to generate augmented_twitter_rumor_dataset.csv.

ontributing
Feel free to open issues or submit pull requests for improvements, bug fixes, or additional features.

Citation
If you use this code in your research, please cite our paper:
[Masnor Alohali]. "Robust Multi-Layer Explainable Rumor Detection with Adversarial Training" [Journal Name], [Year].