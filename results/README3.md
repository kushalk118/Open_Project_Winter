# Assignment 3 – Scalable Quantum Tomography Pipelines

This repository contains the complete and final submission for **Assignment 3: Scalable Quantum Tomography Pipelines**. The objective of this assignment is to design a scalable quantum tomography surrogate, implement serialization and checkpointing, study scalability with increasing qubit counts, and perform ablation studies to analyze architectural design choices.

---

## Repository Structure

assignment-3-quantum-tomography/

- Assignment_3.ipynb  
- scalability_results.csv  
- models.zip  
- README.md  

---

## Assignment_3.ipynb

This notebook contains the full implementation of the assignment, including:

- Serialization helpers implemented using Python `pickle`
- An extendable n-qubit quantum tomography surrogate model
- Construction of normalized complex statevectors
- Pure-state fidelity computation
- Scalability experiments measuring fidelity and runtime
- Visualization of scalability metrics
- Ablation studies on model depth
- Summary, discussion, and conclusions

All required code, outputs, plots, and results are generated within this notebook.

---

## Scalability Study

The scalability study evaluates how the quantum tomography surrogate behaves as the number of qubits increases. Experiments are conducted for multiple qubit counts, and the following metrics are measured:

- Mean fidelity  
- Fidelity standard deviation  
- Mean runtime  

As the number of qubits increases, runtime grows rapidly due to the exponential growth of the Hilbert space dimension. Fidelity remains relatively stable for small numbers of qubits but shows increasing variance as system size grows, indicating reduced expressibility and higher sensitivity to random initialization.

Numerical results from these experiments are stored in `scalability_results.csv`, and corresponding plots (fidelity vs qubits and runtime vs qubits) are generated directly in the notebook.

---

## scalability_results.csv

This CSV file contains numerical results from the scalability experiments with the following fields:

- `n_qubits` – number of qubits  
- `fidelity_mean` – mean fidelity across trials  
- `fidelity_std` – fidelity standard deviation  
- `runtime_mean` – mean runtime per experiment  

This file is used for plotting and further analysis of scalability behavior.

---

## Model Checkpoints (models.zip)

Model checkpoints are stored as serialized `.pkl` files using Python `pickle`.  
For GitHub submission, all checkpoints are compressed into `models.zip`.

After extracting `models.zip`, the directory structure is:

models/
- test_model.pkl  
- (optional additional model checkpoints)

These checkpoints demonstrate correct serialization, saving, and loading of the quantum surrogate model.

---

## How to Extract and Load a Saved Model

Extract the checkpoints:

```bash
unzip models.zip
