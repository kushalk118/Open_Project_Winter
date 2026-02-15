import numpy as np
import torch
import scipy.linalg

def random_density_matrix(dim):
    """Generates a random valid density matrix of size dim x dim."""
    # Ginibre ensemble: G = random complex matrix
    G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    return rho

def quantum_fidelity(rho1, rho2):
    """Computes fidelity F(rho1, rho2) = (Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1))))^2"""
    # Using scipy sqrtm
    sqrt_rho1 = scipy.linalg.sqrtm(rho1)
    # Handle potential small numerical noise making it non-hermitian slightly
    term = sqrt_rho1 @ rho2 @ sqrt_rho1
    fidelity = (np.trace(scipy.linalg.sqrtm(term)).real) ** 2
    return np.clip(fidelity, 0.0, 1.0)

def trace_distance(rho1, rho2):
    """Computes trace distance D(rho1, rho2) = 0.5 * Tr(|rho1 - rho2|)"""
    diff = rho1 - rho2
    # eigenvalues of diff
    evals = np.linalg.eigvalsh(diff) # valid for hermitian difference
    return 0.5 * np.sum(np.abs(evals))

def is_valid_state(rho, tol=1e-6):
    """Checks if rho is Hermitian, PSD, and Trace 1."""
    if not np.allclose(rho, rho.conj().T, atol=tol): return False
    if np.abs(np.trace(rho) - 1.0) > tol: return False
    evals = np.linalg.eigvalsh(rho)
    if np.any(evals < -tol): return False
    return True
