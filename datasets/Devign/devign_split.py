import os
import sys
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

def load_json_data(json_file_path):
    """Load JSON data from file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_to_pickle(df, output_path):
    """Save DataFrame to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)

def balance_and_split_data_old(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split data into train, validation, and test sets with balanced project and target."""
    # Rename 'func' column to 'code'
    df = df.rename(columns={'func': 'code'})
    
    # Normalize the code
    # Using 1 sample for demonstration, you can adjust as needed
    # df = df.iloc[:2]
    # print("Original code:", df['code'][0].replace('\n', ' '))  # Debugging output
    # normalizer = CXXNormalization()
    # df = normalizer.normalization_df(df)
    print("Normalized code:", df['code'][0])  # Debugging output
    
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

def balance_and_split_data(df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split data into train, validation, and test sets with balanced target (0/1) ratio per project.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with 'project', 'target', and 'code' columns
    train_ratio (float): Ratio for training set
    valid_ratio (float): Ratio for validation set
    test_ratio (float): Ratio for test set
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: train_df, valid_df, test_df
    """
    # Rename 'func' column to 'code'
    df = df.rename(columns={'func': 'code'})
    
    # Normalize the code (uncomment if normalization is needed)
    # normalizer = CXXNormalization()
    # df = normalizer.normalization_df(df)
    
    print("Normalized code:", df['code'].iloc[0])  # Debugging output
    
    # Step 1: Balance target (0/1) ratio per project
    balanced_df = pd.DataFrame()
    for project in df['project'].unique():
        project_df = df[df['project'] == project]
        target_counts = project_df['target'].value_counts()
        min_target_count = target_counts.min()  # Get minimum count of target 0 or 1
        
        # Sample equal number of target 0 and target 1
        target_0_df = project_df[project_df['target'] == 0].sample(
            n=min_target_count, 
            random_state=random_state
        )
        target_1_df = project_df[project_df['target'] == 1].sample(
            n=min_target_count, 
            random_state=random_state
        )
        
        # Combine balanced samples for this project
        balanced_project_df = pd.concat([target_0_df, target_1_df])
        balanced_df = pd.concat([balanced_df, balanced_project_df])
    
    print(f"Balanced dataset: {len(balanced_df)} samples, equal target ratio per project")
    
    # Step 2: Create stratification key based on project and target
    balanced_df['stratify_key'] = balanced_df['project'] + '_' + balanced_df['target'].astype(str)
    
    # Step 3: First split - train+valid vs test
    train_valid_df, test_df = train_test_split(
        balanced_df,
        test_size=test_ratio,
        stratify=balanced_df['stratify_key'],
        random_state=random_state
    )
    
    # Step 4: Adjust validation ratio for second split
    valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
    
    # Step 5: Second split - train vs validation
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_ratio_adjusted,
        stratify=train_valid_df['stratify_key'],
        random_state=random_state
    )
        
    # Step 6: Drop the stratification key
    train_df = train_df.drop(columns=['stratify_key'])
    valid_df = valid_df.drop(columns=['stratify_key'])
    test_df = test_df.drop(columns=['stratify_key'])
    
    # Print split sizes and target distribution for verification
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(valid_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print("Target distribution in train set:", train_df['target'].value_counts().to_dict())
    print("Target distribution in validation set:", valid_df['target'].value_counts().to_dict())
    print("Target distribution in test set:", test_df['target'].value_counts().to_dict())
    
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
    # input_json = os.path.join(root_path, 'datasets', 'Devign', 'devign.json')
    # output_directory = os.path.join(root_path, 'datasets', 'Devign')      # Replace with your desired output directory
    main('devign.json', '.')