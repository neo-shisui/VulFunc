import os
import json
import pickle
import pandas as pd
import argparse

def load_json_data(json_file_path):
    """Load JSON data from file."""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

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

def balance_data(df, random_state=42):
    """
    Downsample data to have equal target (0/1) ratio per project.

    Parameters:
    df (pd.DataFrame): Input dataframe with 'project', 'target', and 'func' columns
    random_state (int): Random seed for reproducibility

    Returns:
    pd.DataFrame: Balanced dataframe
    """
    # Rename 'func' column to 'code'
    df = df.rename(columns={'func': 'code'})

    # Balance target (0/1) ratio per project
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

    # Print number of samples per project
    project_counts = balanced_df.groupby('project').size()
    print("Number of samples per project in the balanced dataset:")
    for project, count in project_counts.items():
        print(f"  {project}: {count}")

    print(f"Balanced dataset: {len(balanced_df)} samples, equal target ratio per project")
    print("Target distribution in balanced dataset:", balanced_df['target'].value_counts().to_dict())
    return balanced_df

def show_dataset_statistics(input_json):
    # Load the cleaned dataset
    df = pd.read_json(input_json, orient='records', lines=True)

    # Print dataset statistics
    print("Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Unique projects: {df['project'].nunique()}")
    
    # Count vulnerabilities
    vulnerability_counts = df['target'].value_counts()
    print("\nVulnerability Counts:")
    for label, count in vulnerability_counts.items():
        print(f"{label}: {count} samples")
    
def cleaned_dataset(json_file_path, output_path):
    """Main function to process and downsample Devign data."""
    # Load data
    df = load_json_data(json_file_path)

    # Drop commit column if it exists
    if 'commit_id' in df.columns:
        df = df.drop(columns=['commit_id'])

    # Downsample data
    balanced_df = balance_data(df)

    # Save to JSON file
    save_to_json(balanced_df, output_path)
    print(f"Balanced dataset saved to {output_path}")
    print(f"Total samples: {len(balanced_df)}")

if __name__ == "__main__":
    # handle command line arguments
    parser = argparse.ArgumentParser(description='Process Devign dataset.')
    parser.add_argument('--clean', action='store_true', help='Clean the dataset')
    parser.add_argument('--statistics', action='store_true', help='Show dataset statistics')
    args = parser.parse_args()

    if args.clean:
        input_json = 'devign_original.json'
        output_file = 'devign.json'
        cleaned_dataset(input_json, output_file)
    if args.statistics:
        input_json = 'devign.json'  # Path to the cleaned JSON file
        show_dataset_statistics(input_json)