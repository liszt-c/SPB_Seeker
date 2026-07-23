# SPB-Seeker

---

**SPB-Seeker** (Short Peptide Binder Seeker) is an integrated computational pipeline for the *de novo* design and screening of dual‑target short peptide inhibitors (<10 amino acids). It explores the design of dual‑target short peptide inhibitors against the penicillin‑binding proteins PBP2b and PBP2x of drug‑resistant *Streptococcus pneumoniae*.  

The pipeline combines three representative deep generative models (AFDesign, RFdiffusion, BoltzGen) to maximize chemical space coverage, followed by a multi‑stage computational screening cascade:  
- **Toxicity prediction** using an ESM2‑based classifier to filter hemolytic and cytotoxic peptides.  
- **Molecular docking** (MDockPeP2) for preliminary binding assessment.  
- **Hierarchical molecular dynamics** with MM/PB(GB)SA free‑energy calculations for rigorous validation.  

Diversity analysis via sequence similarity networks (SSN) and UMAP/K‑means clustering further ensures library quality. The final candidates exhibit high predicted affinity and stability toward both targets, providing promising leads for next‑generation antimicrobial therapeutics.  



--- 

## 01 Peptide Generation Scripts

The first folder contains scripts and configuration files for running three representative deep generative models used in the SPB-Seeker pipeline:

- **AFDesign** – gradient‑based hallucination method using AlphaFold2 as a differentiable evaluator.
- **BoltzGen** – all‑atom diffusion model for end‑to‑end structure‑sequence co‑design.
- **RFdiffusion** – backbone diffusion model combined with ProteinMPNN for sequence assignment.


---

### File Overview

| File | Description |
|------|-------------|
| `AFDesign_Binder.py` | Runs AFDesign in binder protocol.  |
| `run_boltzgen.yaml` | YAML configuration for BoltzGen. |
| `run_RFdiffusion_V1.2.py` | Executes RFdiffusion backbone generation followed by ProteinMPNN sequence design. |

---

## Dependencies

### Refer to each tool’s official repository for detailed installation steps.

- Python ≥ 3.9
- PyTorch (with CUDA support recommended)
- [ColabDesign](https://github.com/sokrypton/ColabDesign) (for AFDesign and RFdiffusion wrapper)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) (installed locally)
- [BoltzGen](https://github.com/jostorge/boltzgen) (follow its installation instructions)
- Additional packages: `numpy`, `matplotlib`, `py3Dmol`, `scipy`, `openmm`, etc.

---

## Usage

### 1. AFDesign

Modify the `pdb_filename` and `binder_len` inside the script, then run:
```
python AFDesign_Binder.py
```

Outputs will be saved to `./result/` (binder PDB files and sequence text files).

### 2. BoltzGen

Edit `run_boltzgen.yaml` to point to your target PDB and adjust hotspot residues. Then execute:

```
boltzgen run PBP2B.yaml \
  --output ./PBP2B_run1 \
  --protocol protein-anything \
  --num_designs 1000  \
  --budget 10 \
  --cache /workspace/software/boltzgen_parameters \
  --moldir /workspace/software/huggingface/mols.zip
```
Generated peptides will be written according to BoltzGen’s default output settings.

### 3. RFdiffusion + ProteinMPNN

Set the parameters at the top of `run_RFdiffusion_V1.2.py` (project name, contigs, PDB path, hotspots, etc.), then run:

```
python run_RFdiffusion_V1.2.py
```

The script will first generate backbone designs, then call ProteinMPNN to produce sequences. Results appear under `outputs/<project_name>*/`.

---

## Notes

- All scripts assume that the required third‑party tools (ColabDesign, RFdiffusion, BoltzGen) are installed and accessible via `PYTHONPATH`.

---

## References

- AFDesign: [Kosugi T, Ohue M (2022) Solubility-Aware Protein Binding Peptide Design Using AlphaFold. Biomedicines 10:1626.](https://doi.org/10.3390/biomedicines10071626)
- BoltzGen: [Stark H, Faltings F, Choi M, et al (2025) BoltzGen: Toward Universal Binder Design.](https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1)
- RFdiffusion: [Watson JL, Juergens D, Bennett NR, et al (2023) De novo design of protein structure and function with RFdiffusion. Nature 620:1089–1100.](https://doi.org/10.1038/s41586-023-06415-8)
- ProteinMPNN: [Dauparas J, Anishchenko I, Bennett N, et al (2022) Robust deep learning–based protein sequence design using ProteinMPNN. Science 378:49–56.](https://doi.org/10.1126/science.add2187)


## 02 Molecular Docking Scripts

This folder contains a batch processing script for performing peptide–protein molecular docking using **MDockPeP2**. The script accepts a multi‑FASTA file (typically the output from the generation step) and automatically docks each peptide against a prepared receptor structure (e.g., PBP2b or PBP2x).

---

### File Overview

| File | Description |
|------|-------------|
| `run_mdockpep_v2.5.sh` | Bash script for batch docking using MDockPeP2. |

---

### Dependencies

- **Python 2.7** (required by MDockPeP2)
- http://zougrouptoolkit.missouri.edu/mdockpep/ – follow its installation instructions
- `dos2unix` (optional, for converting Windows‑style line endings)

---

### Usage

1. Prepare a multi‑FASTA file (e.g., `seqs.fasta`) containing peptide sequences.
2. Set the paths at the top of `run_mdockpep_v2.5.sh`:
   - `input_fasta`: path to the FASTA file
   - `receptor_pdb`: path to the receptor PDB file
   - `pep_num`: number of conformations to generate per peptide
   - `CPU_NUM`: number of CPU cores to use
3. Ensure the `python2` executable and MDockPeP2 script are accessible. Edit the script’s internal path to point to your local MDockPeP2 installation.
4. Run:

   ```bash
   bash run_mdockpep_v2.5.sh
   ```

Results for each peptide will be stored in a subfolder named after the peptide ID, containing the docked pose PDB files and MDockPeP2 output logs.

---

### Notes

- MDockPeP2 requires Python 2; ensure compatibility before execution.

---

### References

- MDockPeP2: [Xu X, Zou X (2022) Predicting Protein–Peptide Complex Structures by Accounting for Peptide Flexibility and the Physicochemical Environment. J Chem Inf Model 62:27–39.](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00836)

---

## 03 Toxicity Prediction

Scripts for building and applying an ESM2‑based deep learning model to predict peptide toxicity (hemolysis and cytotoxicity). 
---

### File Overview

| File | Description |
|------|-------------|
| `Seq_to_smiles_single_thread.py` | Converts peptide sequences to SMILES strings using RDKit (single‑threaded for order preservation). |
| `Length_Filtrate.py` | Filters peptides by length (e.g., 3–12 aa) and removes duplicate sequences. |
| `Balance.py` | Balances positive/negative classes via downsampling; supports optional reference dataset merging. |
| `ESM_Pre_V2_2026.4.24_modified.py` | Main training/prediction script: uses ESM2 as a feature extractor + MLP classifier. Supports ablation experiments (different hidden sizes). |

---
### Data Sources

- **Hemolytic activity data**: Collected from the following databases:
  - **HemoPI** – https://doi.org/10.1038/s42003-025-07615-w
  - **APD (Antimicrobial Peptide Database)** – https://doi.org/10.1093/nar/gkaf860
  - **Peptide-Dashboard** – https://doi.org/10.1021/acs.jcim.2c01317
  - **Hemolytik2** – https://doi.org/10.1021/acs.chemrestox.5c00322

- **Cytotoxicity data**: Collected from **ToxiPep** – https://doi.org/10.1016/j.csbj.2025.05.039

---
### Dependencies

- Python ≥ 3.8
- PyTorch (CUDA recommended)
- Transformers (Hugging Face)
- RDKit
- scikit-learn
- pandas, numpy, tqdm
- ESM2 model files (download from [Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D) or use local path)

---

### Usage

#### 1. Data Preprocessing

Run the three preprocessing scripts in order:
```
# (Optional) Step 1: Convert sequences to SMILES for ChemBERTa input
python Seq_to_smiles_single_thread.py

# Step 2: Filter by length and deduplicate
python Length_Filtrate.py

# Step 3: Balance classes
python Balance.py
```

Adjust file paths inside each script as needed.

#### 2. Train the ESM‑based model

```
python ESM_Pre_V2_2026.4.24_modified.py train \
--data ToxiDataset.csv \
--model ./esm2_t6_8M_UR50D \
--hidden_size 512
```
Arguments:
- `--data` : CSV file with `sequence` and `label` columns.
- `--model` : Path or Hugging Face model ID for ESM2.
- `--hidden_size` : Dimension of the MLP hidden layer (used for ablation).

The trained model and performance metrics are saved under `models/<dataset>_<hidden_size>/` and `results/<dataset>_<hidden_size>/`.

#### 3. Predict on new sequences
```
python ESM_Pre_V2_2026.4.24_modified.py pre \
--input new_peptides.csv \
--model_dir models/ToxiDataset_Length_Filtered_512
```

The output CSV will contain `prediction` (probability) and `prediction_binary` columns.

---

### Notes

- The tokenizer and model configuration are saved alongside the weights for easy reloading.
- For ablation studies, change the `--hidden_size` argument and compare results across runs.

---

### References

- ESM2: [Lin et al., Science 2023](https://doi.org/10.1126/science.ade2574)
- RDKit: https://www.rdkit.org/


## 04 Diversity Analysis (SSN)

Scripts for constructing and visualizing a Sequence Similarity Network (SSN) to assess the diversity of generated peptides. 

---

### File Overview

| File | Description |
|------|-------------|
| `ssn_pipeline.py` | Builds the SSN from a multi‑FASTA file of generated peptides. Computes pairwise similarity (Levenshtein or Needleman‑Wunsch), scans thresholds, and selects an optimal cutoff. Outputs a serialized graph (`graph.pkl`) and a threshold‑scan CSV. |
| `plot_ssn.py` | Loads `graph.pkl` and renders an SSN image. Node color encodes generation method; node size encodes degree; edge width encodes similarity weight. |

---

### Dependencies

- Python ≥ 3.8
- NetworkX
- NumPy
- Matplotlib
- Biopython
- python‑Levenshtein (`pip install python-Levenshtein`)

---

### Usage

#### 1. Run the SSN pipeline
```
python ssn_pipeline.py
```

Modify the parameters at the top of the script:
- `FASTA_FILE`: path to the input multi‑FASTA file
- `SIMILARITY_METRIC`: `"levenshtein"` or `"needleman"`
- `THRESHOLD_MIN`, `THRESHOLD_MAX`, `THRESHOLD_STEP`: range and granularity of threshold scanning
- `OUTPUT_DIR`: directory for outputs

Output files:
- `ssn_results/threshold_scan.csv` – metrics at each threshold
- `ssn_results/graph.pkl` – graph object at the optimal threshold

#### 2. Plot the SSN

```
python plot_ssn.py
```

Adjust plotting parameters in the script header. The rendered PDF is saved to `ssn_results/ssn_plot.pdf`.

---

### Notes

- The pipeline automatically extracts `target` and `method` labels from sequence IDs (format: `Target_Method_Index`, e.g., `PBP2B_RFD_0`).
- The plot script uses a non‑interactive Matplotlib backend (`Agg`).

---

### References

- Louvain community detection: [Blondel et al., J. Stat. Mech. 2008](https://doi.org/10.1088/1742-5468/2008/10/P10008)
- NetworkX: [https://networkx.org/](https://networkx.org/)

---
## Note

- The comments in the source code were translated into English with the assistance of DeepSeek.

- For any questions or inquiries, please contact the corresponding author.

---

