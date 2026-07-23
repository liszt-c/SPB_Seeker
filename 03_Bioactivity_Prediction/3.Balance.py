import pandas as pd
import numpy as np
import os

def balance_dataset(input_file, output_file, random_seed=42):
    """
    Balance positive and negative classes in a dataset.
    
    Parameters:
    input_file: Path to input CSV file
    output_file: Path to output CSV file
    random_seed: Random seed for reproducibility
    """
    try:
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # 1. Read CSV file
        df = pd.read_csv(input_file)
        
        print(f"Original row count: {len(df)}")
        print(f"Original columns: {list(df.columns)}")
        
        # 2. Check for label column
        if 'label' not in df.columns:
            print("Error: Dataset does not contain a 'label' column")
            return None
        
        # 3. Count class distribution
        label_counts = df['label'].value_counts()
        print(f"\nClass distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} rows ({count/len(df):.2%})")
        
        # 4. Identify minority and majority classes
        min_label = label_counts.idxmin()
        max_label = label_counts.idxmax()
        min_count = label_counts.min()
        max_count = label_counts.max()
        
        print(f"\nMinority class: Label {min_label} ({min_count} rows)")
        print(f"Majority class: Label {max_label} ({max_count} rows)")
        print(f"Imbalance ratio: {max_count/min_count:.2f}:1")
        
        # 5. Separate minority and majority
        minority_df = df[df['label'] == min_label]
        majority_df = df[df['label'] == max_label]
        
        # 6. Downsample majority class
        if len(majority_df) > len(minority_df):
            print(f"\nDownsampling majority class (Label {max_label})...")
            majority_downsampled = majority_df.sample(n=min_count, random_state=random_seed)
            print(f"Majority class after downsampling: {len(majority_downsampled)}")
        else:
            majority_downsampled = majority_df
        
        # 7. Merge balanced data
        balanced_df = pd.concat([minority_df, majority_downsampled], ignore_index=True)
        
        # 8. Shuffle data
        balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # 9. Save result
        balanced_df.to_csv(output_file, index=False)
        
        # 10. Print statistics
        print(f"\nBalancing complete! Results saved to: {output_file}")
        print(f"Balanced data shape: {balanced_df.shape}")
        
        balanced_counts = balanced_df['label'].value_counts()
        print(f"\nBalanced class distribution:")
        for label, count in balanced_counts.items():
            print(f"  Label {label}: {count} rows ({count/len(balanced_df):.2%})")
        
        return balanced_df
        
    except Exception as e:
        print(f"Error during balancing: {e}")
        import traceback
        traceback.print_exc()
        return None

def balance_with_reference_dataset(input_file, reference_file, output_file, random_seed=42):
    """
    Balance dataset using a reference dataset (e.g., external negative controls).
    
    Parameters:
    input_file: Path to main input CSV file
    reference_file: Path to reference CSV file
    output_file: Path to output CSV file
    random_seed: Random seed for reproducibility
    """
    try:
        np.random.seed(random_seed)
        
        # Read main dataset
        df = pd.read_csv(input_file)
        print(f"Main dataset rows: {len(df)}")
        
        # Read reference dataset
        ref_df = pd.read_csv(reference_file)
        print(f"Reference dataset rows: {len(ref_df)}")
        
        # Check label columns
        if 'label' not in df.columns or 'label' not in ref_df.columns:
            print("Error: One or both datasets lack a 'label' column")
            return None
        
        # Main dataset class distribution
        main_counts = df['label'].value_counts()
        print(f"\nMain dataset class distribution:")
        for label, count in main_counts.items():
            print(f"  Label {label}: {count} rows ({count/len(df):.2%})")
        
        # Reference dataset class distribution
        ref_counts = ref_df['label'].value_counts()
        print(f"\nReference dataset class distribution:")
        for label, count in ref_counts.items():
            print(f"  Label {label}: {count} rows ({count/len(ref_df):.2%})")
        
        # Identify minority and majority classes in main dataset
        min_label = main_counts.idxmin()
        max_label = main_counts.idxmax()
        min_count = main_counts.min()
        max_count = main_counts.max()
        
        print(f"\nMinority class: Label {min_label} ({min_count} rows)")
        print(f"Majority class: Label {max_label} ({max_count} rows)")
        
        minority_df = df[df['label'] == min_label]
        majority_df = df[df['label'] == max_label]
        
        # Downsample majority class
        if len(majority_df) > len(minority_df):
            print(f"\nDownsampling majority class (Label {max_label})...")
            majority_downsampled = majority_df.sample(n=min_count, random_state=random_seed)
            print(f"Majority after downsampling: {len(majority_downsampled)}")
        else:
            majority_downsampled = majority_df
        
        # Merge balanced main dataset
        balanced_main = pd.concat([minority_df, majority_downsampled], ignore_index=True)
        
        # Concatenate with reference dataset
        final_df = pd.concat([balanced_main, ref_df], ignore_index=True)
        
        # Shuffle
        final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Save
        final_df.to_csv(output_file, index=False)
        
        print(f"\nBalancing complete! Results saved to: {output_file}")
        print(f"Final dataset shape: {final_df.shape}")
        
        final_counts = final_df['label'].value_counts()
        print(f"\nFinal class distribution:")
        for label, count in final_counts.items():
            print(f"  Label {label}: {count} rows ({count/len(final_df):.2%})")
        
        return final_df
        
    except Exception as e:
        print(f"Error during balancing: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_dataset_balance(input_file):
    """
    Analyze the balance of a dataset.
    
    Parameters:
    input_file: Path to input CSV file
    """
    try:
        df = pd.read_csv(input_file)
        
        print(f"Dataset analysis report:")
        print(f"Total rows: {len(df)}")
        
        if 'label' not in df.columns:
            print("Error: Dataset does not contain a 'label' column")
            return None
        
        label_counts = df['label'].value_counts().sort_index()
        print(f"\nClass distribution:")
        for label, count in label_counts.items():
            proportion = count / len(df)
            print(f"  Label {label}: {count} rows ({proportion:.2%})")
        
        min_count = label_counts.min()
        max_count = label_counts.max()
        imbalance_ratio = max_count / min_count
        
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print("Warning: Dataset is significantly imbalanced!")
        
        print(f"\nBalancing suggestion:")
        print(f"  Target samples per class after downsampling: {min_count}")
        print(f"  Class to downsample: Label {label_counts.idxmax()}")
        print(f"  Number to downsample: {max_count - min_count}")
        
        return label_counts
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

# Example usage
if __name__ == "__main__":
    input_csv = "Hemolysis_datasets.csv"          # Path to input CSV file
    output_csv = "Hemolysis_datasets_balance.csv" # Path to output CSV file
    
    # Optional: reference dataset path
    # reference_csv = "reference_data.csv"
    
    # 1. Analyze balance
    print("="*50)
    print("Dataset Balance Analysis")
    print("="*50)
    label_stats = analyze_dataset_balance(input_csv)
    
    if label_stats is not None:
        # 2. Perform balancing
        print(f"\n{'='*50}")
        print("Performing Dataset Balancing")
        print("="*50)
        
        balanced_data = balance_dataset(input_csv, output_csv, random_seed=42)
        
        if balanced_data is not None:
            # 3. Verify balance
            print(f"\n{'='*50}")
            print("Post-balance Verification")
            print("="*50)
            
            print(f"\nFirst 10 rows of balanced data:")
            print(balanced_data.head(10))
            
            balanced_counts = balanced_data['label'].value_counts()
            if len(balanced_counts) == 2:
                ratio = balanced_counts.max() / balanced_counts.min()
                if ratio == 1.0:
                    print(f"\n✓ Dataset perfectly balanced (1:1)")
                else:
                    print(f"\n⚠ Dataset nearly balanced ({ratio:.2f}:1)")
            
            # Save statistics report
            stats_file = "balance_statistics.txt"
            with open(stats_file, 'w') as f:
                f.write("Dataset Balancing Statistics Report\n")
                f.write("="*40 + "\n")
                f.write(f"Original dataset: {input_csv}\n")
                f.write(f"Balanced dataset: {output_csv}\n")
                f.write(f"Random seed: 42\n\n")
                
                f.write("Original class distribution:\n")
                for label, count in label_stats.items():
                    f.write(f"  Label {label}: {count} rows ({count/label_stats.sum():.2%})\n")
                
                f.write(f"\nImbalance ratio: {label_stats.max()/label_stats.min():.2f}:1\n\n")
                
                f.write("Balanced class distribution:\n")
                for label, count in balanced_counts.items():
                    f.write(f"  Label {label}: {count} rows ({count/len(balanced_data):.2%})\n")
                
                f.write(f"\nFinal ratio: 1:1\n")
            
            print(f"\nStatistics report saved to: {stats_file}")
    