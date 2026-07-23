#!/bin/bash

# Define input and output directories
input_fasta="./seqs.fasta"  # Path to the input FASTA file
receptor_pdb="./PB2B.pdb"    # Path to the receptor PDB file
work_dir=$(dirname "$input_fasta")  # Working directory derived from input file location
base_name=$(basename "$input_fasta" .fasta)
pep_num="256"    # Number of conformations to generate
CPU_NUM="32"

# Convert FASTA file to Unix line endings
if command -v dos2unix &> /dev/null; then
    dos2unix "$input_fasta"
else
    echo "dos2unix could not be found. Please install it to ensure proper line ending conversion."
    exit 1
fi

# Initialize variables
current_fasta=""
current_pep_id=""

# Function to process a single sequence
process_sequence() {
    local pep_id="$1"
    local fasta_content="$2"
    
    # Create subdirectory using a sanitized peptide ID (remove problematic characters)
    local safe_pep_id=$(echo "$pep_id" | tr -d '\r' | tr -cd '[:alnum:]._-')
    if [[ -z "$safe_pep_id" ]]; then
        safe_pep_id="sequence_$(date +%s%N)" # Generate temporary unique ID if empty
    fi
    local pep_dir="${work_dir}/${safe_pep_id}"
    mkdir -p "$pep_dir"
    
    # Write FASTA content safely using printf to avoid unintended interpretation
    printf "%s" "$fasta_content" > "${pep_dir}/${safe_pep_id}.fasta"
    
    echo "Processing sequence: ${pep_id}"
    echo "Output directory: ${pep_dir}"
    
    # Build command arguments using array to avoid quoting issues
    local cmd_args=(
        python2
        /path/to/mdockpep2.1.py
        --cpu_num "$CPU_NUM"
        --receptor "$receptor_pdb"
        --peptide "${pep_dir}/${safe_pep_id}.fasta"
        --pep_num "$pep_num"
        --out "${pep_dir}/${safe_pep_id}"
    )
    
    # Execute command
    "${cmd_args[@]}"
    local ret=$?
    if [[ $ret -ne 0 ]]; then
        echo "Warning: Processing of sequence '$pep_id' ended with exit code $ret" >&2
    fi
}

# Read and process each line of the FASTA file
while IFS= read -r line || [[ -n $line ]]; do
    # Strip carriage return characters
    line=$(echo "$line" | tr -d '\r')
    
    # Check if line is a header line
    if [[ $line == ">"* ]]; then
        # Process previous sequence if exists
        if [[ -n $current_fasta && -n $current_pep_id ]]; then
            process_sequence "$current_pep_id" "$current_fasta"
        fi
        
        # Update current sequence ID and reset FASTA content
        # Extract ID: remove leading '>', take first space/tab-delimited field
        current_pep_id=$(echo "$line" | sed 's/^>//' | awk '{print $1}')
        current_fasta="${line}"$'\n'
    else
        # Non-header line, append as sequence content
        if [[ -n $line ]]; then
            current_fasta+="${line}"$'\n'
        fi
    fi
done < "$input_fasta"

# Process the last sequence at end of file
if [[ -n $current_fasta && -n $current_pep_id ]]; then
    process_sequence "$current_pep_id" "$current_fasta"
    echo "All sequences processed."
else
    echo "Warning: No valid sequences found in the input file." >&2
fi