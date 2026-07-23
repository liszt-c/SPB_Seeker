import os, time, signal
import sys, random, string, re

import json
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol

from function import get_pdb, run_diffusion
sys.path.append('./')
if 'RFdiffusion' not in sys.path:
    os.environ["DGLBACKEND"] = "pytorch"
    sys.path.append('RFdiffusion')

# Export MKL threading layer manually (environment variable not set automatically)
os.system("export MKL_THREADING_LAYER=GNU")

# Run RFdiffusion
print('Run RFdiffusion')
###############################################################################
###############################################################################

# RFdiffusion parameters
# Project name
Name = "PBP2X"
# Contig definition: chain A, length 7-15
Contigs = "A:7-15"
# Chain(s) considered during computation
Chains = "A"
# Path to the PDB file
PDB = "PBP2B.pdb"
# Number of backbone designs to generate (later used for sequence generation)
Num_Designs = 16
# Hotspot specification (single residues, dot notation)
Hotspot_String = ''       # Range notation, e.g., "A35-44,A70-80"
Hotspot_Dot = 'A273,A306,A308,A309,A332,A336,A385,A485'  # Single residue notation, e.g., "A53,A170"
# ProteinMPNN parameters
# Number of sequences to generate per backbone design
Num_Seqs = 16
# Number of recycling steps
Num_Recycles = 12
# Use initial guess for AlphaFold prediction
initial_guess = True
# Use AlphaFold Multimer v3 parameters for prediction
use_multimer = True

###############################################################################
###############################################################################

name = Name
contigs = Contigs
pdb = PDB
iterations = 100                # Number of iterations for diffusion
num_designs = Num_Designs
visual = "none"                 # Visualization mode: none, image, or interactive
symmetry = "none"               # Symmetry type: none, auto, cyclic, dihedral
order = 1                       # Symmetry order
chains = Chains
add_potential = True            # Add clash penalty between chains

# Process hotspot inputs: combine range notation and dot notation
hotspot_string = Hotspot_String
hotspot_dot = Hotspot_Dot
if hotspot_string:
    # Split range strings into individual residue indices
    ranges = hotspot_string.split(',')
    single_numbers = []
    for range_str in ranges:
        start, end = range_str.split('-')
        start = int(start.lstrip('A'))
        end = int(end.lstrip('A'))
        single_numbers.extend(range(start, end + 1))
    numbers_joined = ',A'.join(str(num) for num in single_numbers)
    hotspot = ''.join(['A', numbers_joined])
    print("Hotspot from ranges:", hotspot)
else:
    hotspot = ''
    print("Hotspot_string is empty, skipping range processing.")

if hotspot_dot:
    if hotspot:  # If previous string is not empty, append with comma
        hotspot = hotspot + ',' + hotspot_dot
    else:
        hotspot = hotspot_dot
    print("Final hotspot:", hotspot)
else:
    print("Hotspot_dot is empty, skipping dot processing.")

# Determine output directory (avoid overwriting existing files)
path = name
while os.path.exists(f"outputs/{path}_0.pdb"):
    path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))

flags = {
    "contigs": contigs,
    "pdb": pdb,
    "order": order,
    "iterations": iterations,
    "symmetry": symmetry,
    "hotspot": hotspot,
    "path": path,
    "chains": chains,
    "add_potential": add_potential,
    "num_designs": num_designs,
    "visual": visual
}

for k, v in flags.items():
    if isinstance(v, str):
        flags[k] = v.replace("'", "").replace('"', '')

contigs, copies = run_diffusion(**flags)

# Run ProteinMPNN
print('Run ProteinMPNN')

num_seqs = Num_Seqs
initial_guess = True
num_recycles = Num_Recycles
use_multimer = False
rm_aa = "C"                     # Amino acids to remove (e.g., cysteine)
mpnn_sampling_temp = 0.1        # Sampling temperature for MPNN

# Wait for AlphaFold parameters download (if needed)
if not os.path.isfile("params/done.txt"):
    print("Downloading AlphaFold params...")
    while not os.path.isfile("params/done.txt"):
        time.sleep(5)

contigs_str = ":".join(contigs)
opts = [
    f"--pdb=outputs/{path}_0.pdb",
    f"--loc=outputs/{path}",
    f"--contig={contigs_str}",
    f"--copies={copies}",
    f"--num_seqs={num_seqs}",
    f"--num_recycles={num_recycles}",
    f"--rm_aa={rm_aa}",
    f"--mpnn_sampling_temp={mpnn_sampling_temp}",
    f"--num_designs={num_designs}"
]
if initial_guess:
    opts.append("--initial_guess")
if use_multimer:
    opts.append("--use_multimer")
opts = ' '.join(opts)
print("Running ProteinMPNN with options:", opts)
os.system(f'python colabdesign/rf/designability_test.py {opts}')