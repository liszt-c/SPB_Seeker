import pandas as pd
from collections import Counter

# 20 standard amino acids (single-letter uppercase)
STANDARD_AAS = list('ARNDCQEGHILKMFPSTWYV')

def compute_aa_frequencies(csv_path, seq_col='sequence'):
    """
    Read a CSV file and compute the frequency of 20 standard amino acids across all sequences.
    
    Parameters:
        csv_path : str   Path to input CSV file
        seq_col  : str   Name of the sequence column (default: 'sequence')
    
    Returns:
        freq_df : pd.DataFrame – DataFrame with two columns: 'AminoAcid' and 'Frequency'
    """
    df = pd.read_csv(csv_path)
    
    # Ensure sequences are strings and convert to uppercase
    seqs = df[seq_col].astype(str).str.upper().tolist()
    
    # Concatenate all sequences into a single string
    all_aas = ''.join(seqs)
    
    # Count occurrences of all characters (including non-standard amino acids)
    raw_counter = Counter(all_aas)
    
    # Keep only standard amino acids; warn about non-standard ones
    non_standard = set(raw_counter.keys()) - set(STANDARD_AAS)
    if non_standard:
        print(f"[Warning] Non-standard amino acids found: {', '.join(sorted(non_standard))}. They are ignored.")
    
    # Build count dictionary for standard amino acids, fill missing with 0
    std_counts = {aa: raw_counter.get(aa, 0) for aa in STANDARD_AAS}
    total = sum(std_counts.values())
    
    if total == 0:
        raise ValueError("No standard amino acids detected in sequences. Please check your data.")
    
    # Compute frequencies
    frequencies = {aa: count / total for aa, count in std_counts.items()}
    
    # Organize into DataFrame
    freq_df = pd.DataFrame({
        'AminoAcid': STANDARD_AAS,
        'Frequency': [frequencies[aa] for aa in STANDARD_AAS]
    })
    
    return freq_df


# ============ Interactive Section ============
if __name__ == '__main__':
    # Method 1: Modify paths directly in the script (recommended for repeated runs)
    input_file = "seq.csv"          #  Path to input CSV file
    output_file = "seq_aa_freq_set.csv"  #  Path to output CSV file
    
    try:
        result = compute_aa_frequencies(input_file)
        result.to_csv(output_file, index=False)
        print(f"Success! Results saved to {output_file}")
        print(result.to_string(index=False))
    except Exception as e:
        print(f"Error: {e}")