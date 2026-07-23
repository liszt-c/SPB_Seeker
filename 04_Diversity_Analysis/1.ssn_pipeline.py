#!/usr/bin/env python3
"""
Sequence Similarity Network (SSN) Pipeline for Short Peptide Diversity Analysis.

Dependencies: pip install biopython networkx numpy matplotlib scikit-learn python-Levenshtein
"""

import re
import sys
import numpy as np
import networkx as nx
from collections import defaultdict
from Bio import SeqIO
from Levenshtein import distance as lev_distance

# ======================== Configurable Parameters ========================
FASTA_FILE = "PBPs_design_2026.7.22.fasta"    # Path to input FASTA file
SIMILARITY_METRIC = "levenshtein"              # "levenshtein" or "needleman"
GAP_OPEN = -2                                  # Needleman-Wunsch gap open penalty
GAP_EXTEND = -1                                # Needleman-Wunsch gap extend penalty
MATCH_SCORE = 1                                # Needleman-Wunsch match score
MISMATCH_SCORE = -1                            # Needleman-Wunsch mismatch score
THRESHOLD_MIN = 0.3                            # Scan start threshold
THRESHOLD_MAX = 0.95                           # Scan end threshold
THRESHOLD_STEP = 0.05                          # Scan step size
OUTPUT_DIR = "./ssn_results/"                  # Output directory path
# ========================================================================

def parse_fasta(filepath):
    """Parse FASTA file and return list of (id, sequence, target, method)."""
    records = []
    for record in SeqIO.parse(filepath, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()
        # Extract target and method from ID, e.g., PBP2B_RFD_0 -> target="PBP2B", method="RFD"
        parts = seq_id.split("_")
        if len(parts) >= 2:
            target = parts[0]
            method = parts[1]
        else:
            target, method = "unknown", "unknown"
        records.append((seq_id, seq, target, method))
    return records

def levenshtein_similarity(seq1, seq2):
    """Compute similarity based on Levenshtein distance: 1 - edit_dist / max_len."""
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 1.0
    dist = lev_distance(seq1, seq2)
    return 1.0 - dist / max_len

def needleman_similarity(seq1, seq2, gap_open=-2, gap_extend=-1,
                         match=1, mismatch=-1):
    """Compute normalized similarity via Needleman-Wunsch global alignment (0~1)."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    score = aligner.score(seq1, seq2)
    max_possible = max(len(seq1), len(seq2)) * match
    if max_possible == 0:
        return 1.0
    normalized = max(0, score / max_possible)
    return min(normalized, 1.0)

def compute_similarity_matrix(records, metric='levenshtein'):
    """Compute pairwise similarity matrix for all sequences."""
    n = len(records)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_mat[i][j] = 1.0
            else:
                if metric == 'levenshtein':
                    s = levenshtein_similarity(records[i][1], records[j][1])
                elif metric == 'needleman':
                    s = needleman_similarity(records[i][1], records[j][1],
                                             GAP_OPEN, GAP_EXTEND,
                                             MATCH_SCORE, MISMATCH_SCORE)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                sim_mat[i][j] = s
                sim_mat[j][i] = s
    return sim_mat

def scan_thresholds(sim_mat, thresholds):
    """Scan multiple thresholds and return network metrics for each."""
    results = []
    n = sim_mat.shape[0]
    for t in thresholds:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if sim_mat[i][j] >= t:
                    G.add_edge(i, j, weight=sim_mat[i][j])

        num_components = nx.number_connected_components(G)
        singletons = sum(1 for node in G.nodes if G.degree[node] == 0)
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G)

        if G.number_of_edges() > 0:
            try:
                communities = nx.community.louvain_communities(G, seed=42)
                modularity = nx.community.modularity(G, communities)
            except Exception:
                modularity = 0.0
        else:
            modularity = 0.0

        results.append({
            'threshold': t,
            'num_components': num_components,
            'singletons': singletons,
            'density': density,
            'avg_clustering': avg_clustering,
            'modularity': modularity,
            'num_edges': G.number_of_edges(),
        })
        print(f"Threshold {t:.2f}: components={num_components}, "
              f"singletons={singletons}, density={density:.4f}, "
              f"clustering={avg_clustering:.4f}, modularity={modularity:.4f}")
    return results

def select_optimal_threshold(results):
    """Select optimal threshold based on modularity and density heuristics."""
    # Prefer thresholds with density > 0.01 and modularity > 0.1
    valid = [r for r in results if r['density'] > 0.01 and r['modularity'] > 0.1]
    if not valid:
        # Fallback: pick threshold closest to density = 0.05
        valid = sorted(results, key=lambda x: abs(x['density'] - 0.05))
    best = max(valid, key=lambda x: x['modularity'])
    return best['threshold']

def build_graph_at_threshold(sim_mat, threshold, records):
    """Build graph at a given threshold, attaching sequence metadata to nodes."""
    n = sim_mat.shape[0]
    G = nx.Graph()
    for idx, (seq_id, seq, target, method) in enumerate(records):
        G.add_node(idx, id=seq_id, seq=seq, target=target, method=method)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_mat[i][j] >= threshold:
                G.add_edge(i, j, weight=sim_mat[i][j])
    return G

def save_indicator_table(results, filename="threshold_scan.csv"):
    """Save threshold scan results to a CSV file."""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'threshold', 'num_components', 'singletons',
            'density', 'avg_clustering', 'modularity', 'num_edges'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Threshold scan table saved to {filename}")

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Parse FASTA
    records = parse_fasta(FASTA_FILE)
    print(f"Loaded {len(records)} sequences.")

    # 2. Compute similarity matrix
    print("Computing similarity matrix...")
    sim_mat = compute_similarity_matrix(records, metric=SIMILARITY_METRIC)

    # 3. Threshold scan
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-6, THRESHOLD_STEP)
    results = scan_thresholds(sim_mat, thresholds)
    save_indicator_table(results, os.path.join(OUTPUT_DIR, "threshold_scan.csv"))

    # 4. Select optimal threshold
    opt_threshold = select_optimal_threshold(results)
    print(f"\nOptimal threshold selected: {opt_threshold:.2f}")

    # 5. Build graph at optimal threshold
    G = build_graph_at_threshold(sim_mat, opt_threshold, records)
    print(f"Graph at threshold {opt_threshold:.2f}: "
          f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # 6. Serialize graph for downstream plotting
    import pickle
    with open(os.path.join(OUTPUT_DIR, "graph.pkl"), 'wb') as f:
        pickle.dump(G, f)
    print("Graph object saved to graph.pkl")