"""Adversarial data augmentation for Twitter rumor detection.

"""

import pandas as pd
import torch
from textattack.augmentation import EmbeddingAugmenter, CharSwapAugmenter
from tqdm import tqdm


def check_device():
    """Check and report GPU availability."""
    is_gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {is_gpu_available}")
    if is_gpu_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    return torch.device("cuda" if is_gpu_available else "cpu")


def load_and_clean_dataset(file_path):
    """Load and preprocess the dataset, removing missing values."""
    df = pd.read_csv(file_path, encoding="utf-8")
    print("Dataset Sample Before Augmentation:")
    print(df.head())
    df = df.dropna(subset=["text", "label"])
    return df


def augment_rumor_texts(df_rumors, embedding_augmenter, char_swap_augmenter):
    """Apply adversarial augmentation to rumor texts."""
    augmented_texts = []
    augmented_labels = []

    for text in tqdm(df_rumors["text"].tolist(), desc="Augmenting Rumor Texts"):
        try:
            # Randomly choose between embedding or character swap augmentation
            if torch.rand(1).item() < 0.5:
                augmented_versions = embedding_augmenter.augment(text)
            else:
                augmented_versions = char_swap_augmenter.augment(text)

            for aug_text in augmented_versions:
                augmented_texts.append(aug_text)
                augmented_labels.append(1)  # Rumor label preserved
        except Exception as e:
            print(f"Skipping text due to error: {e}")

    return pd.DataFrame({"text": augmented_texts, "label": augmented_labels})


def main():
    """Main function to execute adversarial data augmentation."""
    # Check device
    device = check_device()
    print(f"Using device: {device}")

    # Load dataset
    input_file = "combined_twitter_rumor_dataset.csv"
    df = load_and_clean_dataset(input_file)

    # Split into rumors and non-rumors
    df_rumors = df[df["label"] == 1]
    df_non_rumors = df[df["label"] == 0]
    print(f"Original Rumor Samples: {len(df_rumors)}")

    # Initialize augmenters
    embedding_augmenter = EmbeddingAugmenter(
        pct_words_to_swap=0.2, transformations_per_example=2
    )
    char_swap_augmenter = CharSwapAugmenter()

    # Augment rumor texts
    df_augmented = augment_rumor_texts(df_rumors, embedding_augmenter, char_swap_augmenter)
    print(f"Generated Augmented Rumor Samples: {len(df_augmented)}")

    # Save augmented rumors separately
    df_augmented.to_csv("augmented_rumor_only.csv", index=False, encoding="utf-8")

    # Combine with original dataset
    df_final = pd.concat([df, df_augmented]).reset_index(drop=True)
    output_file = "augmented_twitter_rumor_dataset.csv"
    df_final.to_csv(output_file, index=False, encoding="utf-8")

    # Summary
    print("\nDataset Size Comparison:")
    print(f"Before Augmentation: {len(df)} samples")
    print(f"After Augmentation: {len(df_final)} samples (+{len(df_augmented)} new rumors)")
    print(f"\nAdversarial Data Augmentation Completed! New dataset saved as '{output_file}'")


if __name__ == "__main__":
    main()