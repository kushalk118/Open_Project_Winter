# Assignment 2: Final Report
**Date:** January 18, 2026
**Topic:** Quantum State Tomography with Physical Constraints

## 1. Project Overview
This project implements a machine learning approach to Quantum State Tomography (QST). The goal is to reconstruct a quantum density matrix $\rho$ from a set of randomized Pauli measurements (Classical Shadows), enforcing strict physical constraints (Hermitian, Positive Semi-Definite, Unit Trace).

We selected **Track 1 (Classical Shadows)** and utilized a **Transformer-based architecture** to process the measurement sequences.

## 2. Methodology
- **Input Data**: Sequences of (Pauli Basis, Outcome) pairs simulated via the Classical Shadows protocol.
- **Model Architecture**: A Transformer Encoder that aggregates information from the measurement sequence.
- **Constraint Enforcement**: The model outputs a lower triangular matrix $L$, and the density matrix is constructed as:
  $$ \rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)} $$
  This Cholesky-based formulation guarantees that the output is always a valid physical quantum state by design.

## 3. Performance Metrics
The model was evaluated on a held-out test set of 200 random density matrices (single qubit).

| Metric | Description | Result |
| :--- | :--- | :--- |
| **Mean Fidelity** | Measure of similarity between reconstructed and true state ($1.0$ is perfect). | **0.9656** |
| **Mean Trace Distance** | Geometric distance between states ($0.0$ is perfect). | **0.1385** |
| **Inference Latency** | Average time to reconstruct one state on CPU. | **0.2379 ms** |

## 4. Conclusion
The Transformer model successfully learned to reconstruct quantum states from compressed measurement data (100 shots) with high fidelity (>96%). The Cholesky parameterization proved effective in ensuring all physical constraints were met without requiring complex optimization during inference.
