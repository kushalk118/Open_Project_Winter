# Model Working: Transformer for Quantum State Tomography

## Mathematical Logic
The goal is to map a sequence of measurement outcomes $M = \{(P_i, o_i)\}_{i=1}^N$ to a physical density matrix $\rho$.

### Input Representation
Each measurement consists of a Pauli basis $P \in \{X, Y, Z\}$ and an outcome $o \in \{-1, +1\}$.
We flatten this into a single sequence of tokens. For $N$ measurements, input $x \in \mathbb{Z}^N$ where each token represents a unique (Basis, Outcome) pair.

### Architecture: Transformer Encoder
We use a standard Transformer Encoder to process the measurement sequence. The self-attention mechanism allows the model to correlate measurement outcomes across the entire "shadow" to reconstruct the global state.

### Physical Constraints (Cholesky Decomposition)
To ensure $\rho$ is a valid quantum state (Hermitian, Positive Semi-Definite, Unit Trace), we do not predict $\rho$ directly. Instead, we predict a lower triangular matrix $L$:

$$ \rho_{raw} = L L^\dagger $$

This construction guarantees:
1. **Hermiticity**: $(L L^\dagger)^\dagger = (L^\dagger)^\dagger L^\dagger = L L^\dagger$.
2. **Positive Semi-Definiteness**: For any vector $v$, $v^\dagger \rho_{raw} v = v^\dagger L L^\dagger v = ||L^\dagger v||^2 \geq 0$.

Finally, we enforce unit trace by normalization:

$$ \rho = \frac{\rho_{raw}}{\text{Tr}(\rho_{raw})} $$

## Temporal Dynamics / specialized logic
For this implementation (Track 1), we focused on the **Transformer** architecture as requested. The model learns the statistical correlations between Pauli measurements and the underlying quantum state parameters without requiring explicit inversion of the shadow channel (though it likely learns an approximate inversion internally).
