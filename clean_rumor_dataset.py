import pandas as pd
import re

# Load the dataset
file_path = "rumors.csv"  # Path to the original PHEME dataset file
df = pd.read_csv(file_path, encoding="utf-8")

# Display a sample of the original dataset for verification
print("Original Dataset Sample:")
print(df.head())

# Function to clean text by removing URLs, special characters, and extra spaces
# This corresponds to the text normalization step mentioned in section 3.1.2
def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        # Remove URLs and replace with [URL] placeholder as mentioned in the paper
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Keep only alphanumeric characters and basic punctuation
        text = re.sub(r"[^A-Za-z0-9.,!?\'\s]", "", text)
        # Remove extra whitespace and trim
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return ""  # Return empty string for non-string inputs

# Apply the cleaning function to the text column
df["clean_text"] = df.iloc[:, 0].apply(clean_text)

# Remove duplicates and empty rows as part of data cleaning (section 3.1.2)
df_cleaned = df.drop_duplicates(subset=["clean_text"]).dropna().reset_index(drop=True)

# Display a sample of the cleaned dataset
print("\nCleaned Dataset Sample:")
print(df_cleaned.head(10))

# Save the cleaned dataset for further processing
cleaned_file_path = "cleaned_rumors.csv"
df_cleaned.to_csv(cleaned_file_path, index=False, encoding="utf-8")

# Output summary information
print(f"\nCleaned dataset saved as: {cleaned_file_path}")
print(f"Total cleaned tweets: {len(df_cleaned)}")