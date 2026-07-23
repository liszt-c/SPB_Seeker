import csv

def fasta_to_csv(fasta_file, csv_output):
    """
    Convert a FASTA file to CSV format.
    
    Parameters:
        fasta_file: Path to input FASTA file
        csv_output: Path to output CSV file
    """
    
    try:
        sequences = []
        current_header = None
        current_sequence = []
        
        # Read FASTA file
        with open(fasta_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                    
                if line.startswith('>'):  # Header line
                    # Save previous sequence if exists
                    if current_header is not None:
                        sequences.append({
                            'filename': current_header,
                            'sequence': ''.join(current_sequence)
                        })
                    
                    # Start new sequence
                    # Remove leading '>' character
                    current_header = line[1:].strip()
                    current_sequence = []
                else:  # Sequence line
                    current_sequence.append(line)
        
        # Add the last sequence
        if current_header is not None:
            sequences.append({
                'filename': current_header,
                'sequence': ''.join(current_sequence)
            })
        
        if not sequences:
            print("Warning: No sequences found in FASTA file")
            return
        
        # Write CSV file
        with open(csv_output, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['filename', 'sequence']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()
            for seq in sequences:
                writer.writerow(seq)
        
        print(f"Conversion complete! Total sequences processed: {len(sequences)}")
        print(f"CSV file saved to: {csv_output}")
        
    except FileNotFoundError:
        print(f"Error: File not found '{fasta_file}'")
    except Exception as e:
        print(f"Error processing file: {e}")

# Example usage
if __name__ == "__main__":
    # Set file paths
    fasta_file = "ToxiDataset.fasta"  # Path to input FASTA file
    csv_output = "ToxiDataset.csv"    # Path to output CSV file
    
    # Run conversion
    fasta_to_csv(fasta_file, csv_output)