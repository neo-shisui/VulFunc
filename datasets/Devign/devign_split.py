import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
import os

def load_json_data(json_file_path):
    """Load JSON data from file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_to_pickle(df, output_path):
    """Save DataFrame to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)

def balance_and_split_data(df, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_state=42):
    """Split data into train, validation, and test sets with balanced project and target."""
    # Rename 'func' column to 'code'
    df = df.rename(columns={'func': 'code'})
    
    # Create a stratification key based on project and target
    df['stratify_key'] = df['project'] + '_' + df['target'].astype(str)
    
    # First split: train + validation vs test
    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['stratify_key'],
        random_state=random_state
    )
    
    # Adjust validation ratio for second split
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
    
    # Second split: train vs validation
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_ratio_adjusted,
        stratify=train_valid_df['stratify_key'],
        random_state=random_state
    )
    
    # Drop the stratification key
    train_df = train_df.drop(columns=['stratify_key'])
    valid_df = valid_df.drop(columns=['stratify_key'])
    test_df = test_df.drop(columns=['stratify_key'])
    
    return train_df, valid_df, test_df

def main(json_file_path, output_dir):
    """Main function to process Devign data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_json_data(json_file_path)
    
    # Split data
    train_df, valid_df, test_df = balance_and_split_data(df)
    
    # Save to pickle files
    save_to_pickle(train_df, os.path.join(output_dir, 'train.pkl'))
    save_to_pickle(valid_df, os.path.join(output_dir, 'valid.pkl'))
    save_to_pickle(test_df, os.path.join(output_dir, 'test.pkl'))
    
    print(f"Data split and saved:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(valid_df)} samples")
    print(f"Test: {len(test_df)} samples")

if __name__ == "__main__":
    # Example usage
    input_json = "devign.json"  # Replace with your JSON file path
    output_directory = "."      # Replace with your desired output directory
    main(input_json, output_directory)