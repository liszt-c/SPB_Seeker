import pandas as pd

def filter_sequences_by_length(input_file, output_file, length_range="3,10"):
    """
    Filter peptide sequences by length and remove duplicates.
    
    Parameters:
    input_file: Path to input CSV file
    output_file: Path to output CSV file
    length_range: String specifying length range, format "min,max" (e.g., "3,10")
    """
    try:
        # 1. Read CSV file
        df = pd.read_csv(input_file)
        
        print(f"Original row count: {len(df)}")
        print(f"Original columns: {list(df.columns)}")
        print(f"First 5 rows:")
        print(df.head())
        
        # 2. Parse length range
        try:
            min_len, max_len = map(int, length_range.split(','))
            print(f"Filtering length range: {min_len} to {max_len} (inclusive)")
        except ValueError:
            print("Error: Length range format invalid. Expected 'min,max' (e.g., '3,10')")
            return None
        
        if min_len < 1:
            print("Warning: Minimum length should be greater than 0")
        if max_len < min_len:
            print("Error: Maximum length must be >= minimum length")
            return None
        
        # 3. Calculate sequence lengths
        df['seq_length'] = df['sequence'].astype(str).str.len()
        
        # 4. Filter by length range
        before_filter = len(df)
        filtered_df = df[(df['seq_length'] >= min_len) & (df['seq_length'] <= max_len)]
        after_filter = len(filtered_df)
        
        print(f"Before length filter: {before_filter} rows")
        print(f"After length filter: {after_filter} rows")
        print(f"Removed {before_filter - after_filter} rows")
        
        if after_filter == 0:
            print("Warning: No data after filtering!")
            print("Current length distribution:")
            print(df['seq_length'].describe())
            print("\nLength value counts:")
            print(df['seq_length'].value_counts().sort_index().head(20))
            return None
        
        # 5. Remove duplicate sequences (keep first occurrence)
        before_dedup = len(filtered_df)
        dedup_df = filtered_df.drop_duplicates(subset=['sequence'], keep='first')
        after_dedup = len(dedup_df)
        
        print(f"Before deduplication: {before_dedup} rows")
        print(f"After deduplication: {after_dedup} rows")
        print(f"Removed {before_dedup - after_dedup} duplicate sequences")
        
        # 6. Drop temporary length column
        result_df = dedup_df.drop(columns=['seq_length'])
        
        # 7. Save result
        result_df.to_csv(output_file, index=False)
        
        # 8. Statistics summary
        print(f"\nCleaning complete! Results saved to: {output_file}")
        print(f"Final data shape: {result_df.shape}")
        
        # Label distribution if available
        if 'label' in result_df.columns:
            label_counts = result_df['label'].value_counts()
            print(f"\nLabel distribution:")
            for label, count in label_counts.items():
                print(f"  Label {label}: {count} rows ({count/len(result_df):.2%})")
        
        # Length statistics
        print(f"\nFinal sequence length distribution:")
        lengths = result_df['sequence'].str.len()
        print(f"  Shortest: {lengths.min()} amino acids")
        print(f"  Longest: {lengths.max()} amino acids")
        print(f"  Mean: {lengths.mean():.2f} amino acids")
        print(f"  Median: {lengths.median()} amino acids")
        
        return result_df
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    input_csv = "ToxiDataset.csv"           # Path to input CSV file
    output_csv = "ToxiDataset_Length_Filtered.csv"  # Path to output CSV file
    
    # Define length range: filter sequences between 3 and 12 amino acids
    length_range = "3,12"
    
    filtered_data = filter_sequences_by_length(input_csv, output_csv, length_range)
    
    if filtered_data is not None:
        print("\nFirst 10 rows of processed data:")
        print(filtered_data.head(10))