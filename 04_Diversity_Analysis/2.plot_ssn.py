#!/usr/bin/env python3
"""
Plot Sequence Similarity Network (SSN) from a pre-computed graph pickle.

Output: PDF vector graphic.
"""
# Use non-interactive backend for headless servers
import matplotlib
matplotlib.use('Agg')

import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================== Configurable Parameters ========================
GRAPH_PKL = "./ssn_results/graph.pkl"          # Path to graph pickle from pipeline
OUTPUT_PDF = "./ssn_results/ssn_plot.pdf"      # Path to output PDF

# --- Layout parameters ---
LAYOUT_SEED = 20                               # Random seed for reproducibility
LAYOUT_K = 0.99                                # spring_layout k (larger = more spread)
LAYOUT_ITERATIONS = 1000                       # Number of layout iterations

# --- Node style ---
NODE_SIZE_MULTIPLIER = 180                     # Node size = degree * multiplier + base
NODE_SIZE_BASE = 60                            # Base node size
NODE_ALPHA = 0.45                              # Node transparency

# --- Edge style ---
EDGE_WIDTH_MIN = 3                             # Minimum edge width
EDGE_WIDTH_MAX = 8                             # Maximum edge width (scaled by weight)
EDGE_ALPHA = 0.7                               # Edge transparency

# --- Color map (method -> color) ---
METHOD_COLOR_MAP = {
    'AFD': '#E63946',                          # Red
    'RFD': '#457B9D',                          # Blue
    'BG':  '#2A9D8F',                          # Green
}
DEFAULT_COLOR = "#323030"                      # Fallback color for unknown methods

# --- Legend parameters ---
LEGEND_FONTSIZE = 10
LEGEND_LOC = 'best'

# --- Figure parameters ---
FIG_SIZE = (12, 10)
DPI = 600
# ==========================================================================

def load_graph(pkl_path):
    """Load graph object from pickle file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Graph file not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)
    return G

def get_method_color(method):
    """Return color for a given generation method."""
    return METHOD_COLOR_MAP.get(method.upper(), DEFAULT_COLOR)

def draw_ssn(G, output_pdf):
    """Render the SSN graph and save as PDF."""
    # Extract node attributes
    methods = [data['method'] for _, data in G.nodes(data=True)]
    degrees = dict(G.degree())

    # Node colors and sizes
    node_colors = [get_method_color(m) for m in methods]
    node_sizes = [NODE_SIZE_BASE + NODE_SIZE_MULTIPLIER * deg for deg in degrees.values()]

    # Edge widths (linearly scaled by weight)
    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
    if edge_weights:
        w_min, w_max = min(edge_weights), max(edge_weights)
            # Avoid division by zero
        if w_max - w_min > 1e-6:
            edge_widths = [
                EDGE_WIDTH_MIN + (w - w_min) / (w_max - w_min) * (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN)
                for w in edge_weights
            ]
        else:
            edge_widths = [EDGE_WIDTH_MIN] * len(edge_weights)
    else:
        edge_widths = []

    # Compute layout
    pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=LAYOUT_K, iterations=LAYOUT_ITERATIONS)

    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=EDGE_ALPHA,
        edge_color='grey',
        ax=ax
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=NODE_ALPHA,
        linewidths=0.5,
        edgecolors='black',
        ax=ax
    )

    ax.set_axis_off()

    # Build legend
    handles = []
    for method, color in METHOD_COLOR_MAP.items():
        handle = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=color, markersize=10,
                            label=method)
        handles.append(handle)
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOC, framealpha=0.8)

    # Title (show approximate threshold from first edge weight)
    if G.number_of_edges() > 0:
        first_weight = list(G.edges(data=True))[0][2]['weight']
        ax.set_title(f"Sequence Similarity Network\nThreshold ≈ {first_weight:.2f}", fontsize=14)
    else:
        ax.set_title("Sequence Similarity Network (No edges)", fontsize=14)

    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_pdf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"SSN plot saved to {output_pdf}")

if __name__ == "__main__":
    if not os.path.exists(GRAPH_PKL):
        print(f"Error: Graph file '{GRAPH_PKL}' not found. Please run the SSN pipeline first.")
        exit(1)

    G = load_graph(GRAPH_PKL)
    print(f"Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    draw_ssn(G, OUTPUT_PDF)