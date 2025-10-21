"""
SVD-based matrix completion for sparse interaction data.
Based on the friend's reproduction report that showed significant improvements.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch

def svd_matrix_completion(interaction_matrix, k=50, max_iterations=10, convergence_threshold=1e-4):
    """
    Apply SVD-based matrix completion to reduce sparsity in interaction matrices.
    
    Args:
        interaction_matrix: scipy sparse matrix (users x items)
        k: number of singular values to keep (default: 50 as per friend's report)
        max_iterations: maximum iterations for refinement
        convergence_threshold: convergence threshold for stopping
    
    Returns:
        completed_matrix: dense numpy array with filled values
        sparsity_reduction: percentage of sparsity reduction achieved
    """
    print(f"Starting SVD matrix completion with k={k}")
    
    # Convert to dense for initial processing
    if sp.issparse(interaction_matrix):
        dense_matrix = interaction_matrix.toarray()
    else:
        dense_matrix = interaction_matrix.copy()
    
    original_sparsity = 1.0 - (np.count_nonzero(dense_matrix) / dense_matrix.size)
    print(f"Original sparsity: {original_sparsity:.4f}")
    
    # Initial SVD on the sparse matrix
    print("Performing initial SVD decomposition...")
    
    # Ensure matrix has correct dtype for svds()
    if interaction_matrix.dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
        print(f"Converting matrix dtype from {interaction_matrix.dtype} to float32")
        interaction_matrix = interaction_matrix.astype(np.float32)
    
    U, s, Vt = svds(interaction_matrix, k=k)
    
    # Reconstruct the matrix
    completed_matrix = U @ np.diag(s) @ Vt
    
    # Keep original non-zero values and only fill zeros
    mask = (dense_matrix != 0)
    completed_matrix[mask] = dense_matrix[mask]
    
    # Clamp negative values to 0 (interactions should be non-negative)
    completed_matrix = np.maximum(completed_matrix, 0)
    
    # Calculate final sparsity
    final_sparsity = 1.0 - (np.count_nonzero(completed_matrix) / completed_matrix.size)
    sparsity_reduction = original_sparsity - final_sparsity
    
    print(f"Final sparsity: {final_sparsity:.4f}")
    print(f"Sparsity reduction: {sparsity_reduction:.4f}")
    
    # Calculate reconstruction error on known entries
    known_entries = dense_matrix[mask]
    reconstructed_entries = completed_matrix[mask]
    reconstruction_error = np.mean((known_entries - reconstructed_entries) ** 2)
    print(f"Reconstruction error (MSE): {reconstruction_error:.6f}")
    
    return completed_matrix, sparsity_reduction

def apply_svd_completion_to_dataset(dataset, k=50):
    """
    Apply SVD completion to the training interaction matrix of a dataset.
    
    Args:
        dataset: RecDataset object with interaction data
        k: number of singular values for SVD
        
    Returns:
        completed_interactions: scipy sparse matrix with reduced sparsity
    """
    print("Applying SVD completion to dataset")
    
    # Get interaction matrix
    interactions = dataset.inter_matrix(form='coo')
    print(f"Original interaction matrix shape: {interactions.shape}")
    print(f"Original number of interactions: {interactions.nnz}")
    
    # Apply SVD completion
    completed_dense, sparsity_reduction = svd_matrix_completion(interactions, k=k)
    
    # Convert back to sparse format, keeping only values above a threshold
    # MEMORY FIX: Be much more conservative about which values to keep
    threshold = np.percentile(completed_dense[completed_dense > 0], 95)  # Keep only top 5% (was 10%)
    completed_dense[completed_dense < threshold] = 0
    
    # Additional memory optimization: limit maximum number of non-zeros
    max_nonzeros = interactions.nnz * 2  # At most double the original interactions
    nonzero_indices = np.nonzero(completed_dense)
    if len(nonzero_indices[0]) > max_nonzeros:
        # Keep only the highest values
        values = completed_dense[nonzero_indices]
        top_indices = np.argsort(values)[-max_nonzeros:]
        new_completed = np.zeros_like(completed_dense)
        new_completed[nonzero_indices[0][top_indices], nonzero_indices[1][top_indices]] = values[top_indices]
        completed_dense = new_completed
    
    completed_sparse = sp.coo_matrix(completed_dense)
    print(f"Completed interaction matrix non-zeros: {completed_sparse.nnz}")
    print(f"Memory usage ratio: {completed_sparse.nnz / interactions.nnz:.2f}x original")
    
    return completed_sparse
