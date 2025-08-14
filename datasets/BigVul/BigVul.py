import json
import pickle
import pandas as pd
import argparse
from collections import Counter

def create_binary_dataset(df):
    """
    Create a binary classification dataset where func_before is labeled as vulnerable (1)
    and func_after is labeled as not vulnerable (0).
    
    Parameters:
    df (pd.DataFrame): The BigVul dataset DataFrame
    
    Returns:
    pd.DataFrame: Binary classification dataset
    """
    samples = []
    for _, row in df.iterrows():
        samples.append({'code': row['func_before'], 'label': 1})
        samples.append({'code': row['func_after'], 'label': 0})
    binary_df = pd.DataFrame(samples)
    return binary_df

def create_multi_dataset(df, N=5):
    """
    Create a multi-class classification dataset focusing on the top N CWE types.
    Balance the dataset by sampling an equal number of samples from each class.
    
    Parameters:
    df (pd.DataFrame): The BigVul dataset DataFrame
    N (int): Number of top CWE types to consider
    
    Returns:
    pd.DataFrame: Multi-class classification dataset
    """
    # Get top N CWE types
    cwe_counts = df['cwe_id'].value_counts()
    top_cwe = cwe_counts.head(N).index.tolist()
    cwe_to_label = {cwe: i+1 for i, cwe in enumerate(top_cwe)}
    # 0 for no vulnerability

    samples = []
    for _, row in df.iterrows():
        if row['cwe_id'] in top_cwe:
            label = cwe_to_label[row['cwe_id']]
            samples.append({'code': row['func_before'], 'label': label})
        samples.append({'code': row['func_after'], 'label': 0})

    multi_df = pd.DataFrame(samples)

    # Balance the dataset
    label_counts = multi_df['label'].value_counts()
    M = min(label_counts)
    balanced_samples = []
    for label in label_counts.index:
        label_df = multi_df[multi_df['label'] == label]
        sampled_df = label_df.sample(n=M, random_state=42)
        balanced_samples.append(sampled_df)
    balanced_df = pd.concat(balanced_samples)
    return balanced_df

def save_to_json(df, output_path):
    """Save DataFrame to a JSON file as a single array of records.
    
    Args:
        df: pandas DataFrame to save.
        output_path: Path to the output JSON file.
    """
    try:
        # Save DataFrame as a single JSON array (not JSON Lines)
        df.to_json(output_path, orient='records', indent=2)
        print(f"Successfully saved DataFrame to {output_path}")
    except Exception as e:
        print(f"Error saving DataFrame to JSON: {e}")
        raise

def clean_dataset():
    # Load the JSON dataset (replace 'bigvul.json' with your actual file path)
    with open('MSR_data_cleaned.json', 'r') as file:
        data = json.load(file)

    # Convert dictionary to list of records
    if isinstance(data, dict):
        records = [{"id": key, **value} for key, value in data.items()]
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(data)

    # Print all available columns to verify
    # print("Available columns:", df.columns.tolist())

    # Check for the requested columns, adjusting for known naming
    columns_mapping = {
        'cwd_id': 'CWE ID',            # Confirmed from sample
        'target': 'vul',               # Adjust if the actual name differs
        'project': 'project',          # Adjust if the actual name differs
        'lang': 'lang',
        'func_before': 'func_before',  # Adjust if the actual name differs
        'func_after': 'func_after',    # Adjust if the actual name differs
    }

    # Filter columns that exist in the DataFrame
    columns_to_keep = [columns_mapping[col] for col in columns_mapping if columns_mapping[col] in df.columns]
    if not columns_to_keep:
        print("Error: None of the specified columns exist. Available columns:", df.columns.tolist())
    else:
        # Keep only the specified columns
        df_cleaned = df[columns_to_keep]
        
        # Rename columns to match your desired names
        reverse_mapping = {v: k for k, v in columns_mapping.items() if v in df_cleaned.columns}
        df_cleaned = df_cleaned.rename(columns=reverse_mapping)
        
        # Specifically rename 'label' to 'target' if it exists
        if 'label' in df_cleaned.columns:
            df_cleaned = df_cleaned.rename(columns={'label': 'target'})
        
        # Drop rows with empty 'cwd_id'
        df_cleaned.drop(df_cleaned[df_cleaned['cwd_id'] == ''].index, inplace=True)

        # Save the cleaned dataset
        save_to_json(df_cleaned, 'bigvul.json')
        # df_cleaned.to_json('bigvul.json', orient='records', lines=True)
        print("Cleaned dataset saved to 'bigvul.json'")
        print("\nFirst few rows of cleaned data:\n", df_cleaned.head())
    
def show_dataset_statistics():
    # Load the cleaned dataset from JSON
    df = pd.read_json('bigvul.json', orient='records')

    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique CWE IDs: {df['cwd_id'].nunique()}")
    print(f"Unique projects: {df['project'].nunique()}")
    
    # Count vulnerabilities
    vulnerability_counts = df['target'].value_counts()
    print("\nVulnerability Counts:")
    for label, count in vulnerability_counts.items():
        print(f"{label}: {count} samples")
    
    # Print top 10 CWE types
    # Remove record non-vulnerable samples
    df = df[df['target'] == 1]
    top_cwe = df['cwd_id'].value_counts().head(10)
    print("\nTop 10 CWE Types:")
    print(top_cwe)

if __name__ == "__main__":
    # Handle param for clean dataset or statistics
    parser = argparse.ArgumentParser(description='Process BigVul dataset.')
    parser.add_argument('--clean', action='store_true', help='Clean the dataset')
    parser.add_argument('--statistics', action='store_true', help='Show dataset statistics')
    args = parser.parse_args()

    if args.clean:
        clean_dataset()

    if args.statistics:
        show_dataset_statistics()