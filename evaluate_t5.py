import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer
import transformers
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Transformers version: {transformers.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load test dataset
logger.info("Loading test dataset...")
test_df = pd.read_csv("test_dataset.csv")
logger.info(f"Test set loaded with {len(test_df)} rows")
logger.info(f"Test DataFrame head:\n{test_df.head()}")
logger.info(f"Missing values in test_df:\n{test_df.isnull().sum()}")

# Ensure all text columns are strings
test_df["text_input"] = test_df["text_input"].astype(str)
test_df["text_output"] = test_df["text_output"].astype(str)

# Convert to Hugging Face Dataset
def create_hf_dataset(df):
    return Dataset.from_dict({
        "input_text": df["text_input"].tolist(),
        "target_text": df["text_output"].tolist()
    })

test_dataset = create_hf_dataset(test_df)
logger.info(f"Test dataset created with {len(test_dataset)} examples")
logger.info(f"First few input_text samples: {test_dataset['input_text'][:5]}")
logger.info(f"First few target_text samples: {test_dataset['target_text'][:5]}")

# Load T5 tokenizer and model
logger.info("Loading T5 tokenizer and model...")
t5_model_name = "./t5_results/best_model"
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
model.to(device)

# Tokenize function
def tokenize_t5_function(examples):
    logger.info(f"Tokenizing batch with {len(examples['input_text'])} examples")
    inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["target_text"],
            max_length=8,
            padding="max_length",
            truncation=True
        )
    inputs["labels"] = targets["input_ids"]
    inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in inputs["labels"]
    ]
    return inputs

# Tokenize test dataset
logger.info("Tokenizing test dataset...")
test_dataset = test_dataset.map(tokenize_t5_function, batched=True, batch_size=16)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Custom Trainer
class T5GenerationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t5_tokenizer = tokenizer

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=8,
                num_beams=2
            )
            if prediction_loss_only:
                return (None, None, None)
            labels = inputs.get("labels", None)
            return (None, generated_ids, labels)

# Metrics function
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    logger.info(f"eval_pred predictions shape: {predictions.shape}")
    logger.info(f"eval_pred labels shape: {labels.shape}")
    logger.info(f"Raw predictions sample: {predictions[:5]}")  # Log raw predictions

    # Filter invalid token IDs
    valid_range = range(0, tokenizer.vocab_size)  # 0 to 32,000 for T5-small
    predictions = np.where(
        (predictions >= 0) & (predictions < tokenizer.vocab_size),
        predictions,
        tokenizer.pad_token_id  # Replace invalid IDs with pad token
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_norm = [pred.strip().lower() for pred in decoded_preds]
    label_norm = [label.strip().lower() for label in decoded_labels]

    for i in range(min(5, len(pred_norm))):
        logger.info(f"Example {i}: Prediction='{pred_norm[i]}', Actual='{label_norm[i]}'")

    true_predictions = [1 if "true" in pred else 0 for pred in pred_norm]
    true_labels = [1 if "true" in label else 0 for label in label_norm]

    accuracy = accuracy_score(true_labels, true_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='binary', zero_division=0)
    conf_mat = confusion_matrix(true_labels, true_predictions)

    if conf_mat.size == 4:
        tn, fp, fn, tp = conf_mat.ravel()
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Initialize trainer for evaluation
logger.info("Initializing T5 trainer for evaluation...")
trainer = T5GenerationTrainer(
    model=model,
    compute_metrics=compute_metrics
)

# Evaluate
logger.info("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
logger.info(f"T5-Small Test Results: {test_results}")

accuracy = test_results.get("eval_accuracy", 0) * 100
f1 = test_results.get("eval_f1", 0) * 100

logger.info("Final Model Comparison on Twitter Rumor Dataset")
logger.info("RoBERTa Accuracy: 95.84% (Baseline)")
logger.info(f"XLNet Accuracy: 82.38% | F1 Score: 61.65%")
logger.info(f"T5-Small Accuracy: {accuracy:.2f}% | F1 Score: {f1:.2f}%")