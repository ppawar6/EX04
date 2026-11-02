# EX04 – Inventory Control System GNN Analysis

This project adapts the SysML GNN pipeline to analyze a normalized graph generated from the Inventory Control System specification XML.

## Environment

- Python 3.11
- NetworkX
- PyTorch
- PyTorch Geometric
- Transformers
- scikit-learn
- pandas
- NumPy
- Plotly
- PyVis

## Input

The GNN analysis uses:

`inventory_handoff.graphml`

This GraphML file is produced from the specification XML by the XML adapter and contains the normalized nodes and relationships used by the analysis pipeline.

## Install Dependencies

```bash
pip install networkx torch torch-geometric transformers scikit-learn pandas numpy plotly pyvis 