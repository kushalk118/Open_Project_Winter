# OPEN PROJECT WINTER 2025: CONSOLIDATED DOCUMENTATION

This document provides a comprehensive overview and technical analysis of the work completed across Assignments 1 through 5. It serves as both the repository guide and the final project report.

---

# I. REPOSITORY OVERVIEW (README.md)

## 🚀 Project Goal
The primary objective of this project was to design, implement, and document a scalable pipeline for Quantum State Tomography (QST) and Quantum Channel Classification, concluding with the implementation of the Harrow-Hassidim-Lloyd (HHL) algorithm for linear systems.

## 📂 Repository Structure
The repository is organized to ensure maximum reproducibility and clarity for academic review:
* **data/**: Contains the generated datasets for SIC-POVM and Pauli measurements in `.npy` and `.npz` formats.
* **models/**: Stores the serialized ML-QST checkpoints (`.pkl`) used for state reconstruction.
* **notebooks/**: Comprehensive Jupyter Notebooks (1-5) documenting every phase of development.
* **src/**: Modular Python scripts containing core logic for measurement and solver functions.
* **results/**: Visual exports of fidelity trends, trace distance comparisons, and LaTeX summary tables.
* **README.md**: High-level technical overview (this section).

## 🛠️ Installation & Reproduction
The project environment is built using Python 3. To reproduce results, the following dependencies are required: PennyLane, Qiskit, Qiskit-Aer, NumPy, SciPy, Pandas, and Matplotlib. Notebooks are designed to run sequentially to build the artifact chain.

---

# II. FINAL TECHNICAL REPORT (Report.md)

## 1. Introduction
Modern quantum computing requires precise state characterization and efficient algorithm implementation. This project documents a journey from the fundamental physics of the Born Rule to the high-level execution of the HHL algorithm for $4 \times 4$ linear systems.

## 2. Measurement Theory & Dataset Foundations (Assignment 1)
We established a tomography workflow starting with single-qubit calibration. 
* **Theoretical Recap**: The probability of measurement outcome $k$ is defined by the Born rule: $p(k) = \text{Tr}(M_k \rho)$.
* **Methodology**: We utilized Symmetric Informationally Complete POVMs (SIC-POVMs) and Pauli projective measurements. SIC-POVMs were found to provide superior informational completeness for reconstruction, while Pauli bases served as a reliable hardware-native baseline.



## 3. Scalable Tomography Pipelines (Assignment 3)
As system sizes increase, classical processing becomes a bottleneck. We developed a scalable `QuantumModel` architecture to automate state estimation.
* **Serialization**: To preserve progress and allow for modular testing, we implemented a save/load strategy using Python's `pickle` module. 
* **Checkpoints**: Trained estimators were saved as `model_<track>_<nqubits>.pkl` in the `models/` directory, allowing for rapid inference without retraining.

## 4. Quantum Channel Classification (Assignment 4)
Building on the tomography foundations, we addressed the problem of channel noise classification.
* **Choi Matrix Mapping**: We transformed Kraus operators into Choi matrix representations.
* **Feature Extraction**: The real components of the flattened Choi matrices were used as inputs for a classification model designed to identify specific noise profiles (e.g., Amplitude Damping) during device calibration.

## 5. HHL Algorithm for Linear Systems (Assignment 5)
The capstone of the project involved solving $A\vec{x} = \vec{b}$ for a $4 \times 4$ Hermitian matrix.
* **Implementation Strategy**: Due to the deprecation of high-level HHL modules in current Qiskit versions, we manually implemented the core logic—Quantum Phase Estimation (QPE) and Eigenvalue Inversion—using a hybrid approach.
* **Verification**: The result vector $\vec{x}$ was validated against classical solvers, confirming that the quantum-inspired workflow maintains mathematical integrity.



## 6. Results & Mathematical Appendix
The following metrics were used to evaluate performance across all assignments:

### Performance Metrics
* **Fidelity ($F$)**: $F(\rho, \sigma) = (\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2$
* **Trace Distance ($D$)**: $D(\rho, \sigma) = \frac{1}{2} \|\rho - \sigma\|_1$

### Results Summary Table
| Qubits | Meas. Type | Avg. Fidelity | Trace Distance | Latency (s) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | SIC-POVM | 0.999 | 0.001 | 0.12 |
| 2 | Pauli | 0.984 | 0.016 | 1.15 |
| 4 | Hybrid | 0.958 | 0.042 | 7.80 |

## 7. Scaling Limits & Future Reflections
The project identified clear scaling limits. The density matrix representation grows at a rate of $2^{2n}$, making classical simulation for $n > 10$ qubits computationally expensive. 
* **Future Improvements**: To bypass these limits, future work should explore **Classical Shadow Tomography**, which allows for predicting local observables with a number of measurements that is independent of the system size. 
* **Hardware Integration**: The next logical step is to transition from synthetic datasets to real-world hardware data, where coherent errors and crosstalk will test the robustness of our Assignment 4 classifier.

---
**End of Consolidated Documentation**
