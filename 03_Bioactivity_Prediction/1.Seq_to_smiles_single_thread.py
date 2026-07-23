import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def sequence_to_smiles(sequence):
    """Convert an amino acid sequence to a SMILES string."""
    try:
        # Create molecule from sequence
        mol = Chem.MolFromSequence(sequence)
        if mol is None:
            return None
        # Convert to SMILES (without explicit hydrogens)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(f"Error converting sequence '{sequence[:20]}...': {str(e)}")
        return None

# Input and output file paths
input_csv = "Hemolysis_datasets_filtered_length.csv"  # Path to input CSV file
output_csv = "Hemolysis_datasets_smiles_nonH.csv"     # Path to output CSV file

print(f"Reading data from: {input_csv}")
df = pd.read_csv(input_csv)
sequences = df['sequence'].tolist()
total_sequences = len(sequences)

print("Starting single-thread conversion (ensuring correct order)...")

# Sequential processing to maintain order
smiles_results = []
failed_sequences = []

# Progress bar with tqdm
for i, seq in enumerate(tqdm(sequences, desc="Converting sequences", total=total_sequences)):
    smiles = sequence_to_smiles(seq)
    smiles_results.append(smiles)
    
    if smiles is None:
        failed_sequences.append({
            'index': i,
            'sequence': seq,
            'length': len(seq)
        })

# Append results to DataFrame
df['smiles'] = smiles_results

# Check conversion success rate
success_count = df['smiles'].notna().sum()
success_rate = success_count / total_sequences * 100
print(f"Conversion complete: {success_count}/{total_sequences} ({success_rate:.2f}%) successful")

# Report failed conversions if any
if failed_sequences:
    print(f"\nFailed sequences ({len(failed_sequences)} total):")
    for fail in failed_sequences[:10]:  # Show first 10 failures
        seq_preview = fail['sequence'] if len(fail['sequence']) <= 30 else fail['sequence'][:27] + "..."
        print(f"  Row {fail['index']+1}: length={fail['length']}, sequence='{seq_preview}'")
    
    if len(failed_sequences) > 10:
        print(f"  ...and {len(failed_sequences)-10} other failed sequences")

# Save results
df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")

# Optionally save failed sequences to a separate file
if failed_sequences:
    failed_df = pd.DataFrame(failed_sequences)
    failed_csv = "failed_conversions.csv"
    failed_df.to_csv(failed_csv, index=False)
    print(f"Failed sequences saved to {failed_csv}")