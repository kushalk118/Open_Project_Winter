# Replication Guide

## 1. Environment Setup
Ensure you have Python 3.8+ installed.
```bash
cd QCG_Assignment2
pip install -r requirements.txt
```

## 2. Dataset Generation
The dataset is generated on-the-fly during training by `src.dataset.ShadowDataset`.
- **Logic**: Random density matrices are generated using the Ginibre ensemble. Single-qubit Pauli measurements (X, Y, Z) are simulated.
- **Parameters**: Default is 1000 samples, 100 measurements per state.

## 3. Training
Execute the training pipeline:
```bash
python -m src.train
```

This script will:
1. Generate synthetic data.
2. Train the CholeskyTomographyNet for 10 epochs (default).
3. Save the model to `outputs/model.pth`.
4. Run inference on a test set and report Mean Fidelity and Trace Distance.

## 4. Expected Results
On a single-qubit task with 100 measurements per state:
- **Mean Fidelity**: > 0.95
- **Mean Trace Distance**: < 0.10
- **Training Time**: < 1 minute on CPU.
