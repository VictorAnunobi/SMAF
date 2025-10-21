"""
Fusion similarity for improved graph construction.
Combines Pearson correlation and cosine similarity as per friend's reproduction.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

def pearson_similarity(embeddings):
    """
    Calculate Pearson correlation coefficient between embeddings.
    
    Args:
        embeddings: tensor of shape (N, D) where N is number of nodes, D is embedding dimension
        
    Returns:
        similarity_matrix: tensor of shape (N, N) with Pearson correlations
    """
    # Normalize embeddings to zero mean
    embeddings_centered = embeddings - embeddings.mean(dim=1, keepdim=True)
    
    # Calculate correlation matrix
    # Cov(X,Y) = E[(X-μX)(Y-μY)] = (X-μX)^T(Y-μY) / (n-1)
    cov_matrix = torch.mm(embeddings_centered, embeddings_centered.t()) / (embeddings.shape[1] - 1)
    
    # Calculate standard deviations
    std_vector = torch.sqrt(torch.diag(cov_matrix))
    std_matrix = torch.outer(std_vector, std_vector)
    
    # Avoid division by zero
    std_matrix = torch.clamp(std_matrix, min=1e-8)
    
    # Pearson correlation = Cov(X,Y) / (σX * σY)
    pearson_matrix = cov_matrix / std_matrix
    
    # Clamp to [-1, 1] range
    pearson_matrix = torch.clamp(pearson_matrix, min=-1.0, max=1.0)
    
    return pearson_matrix

def cosine_similarity_torch(embeddings):
    """
    Calculate cosine similarity between embeddings using PyTorch.
    
    Args:
        embeddings: tensor of shape (N, D)
        
    Returns:
        similarity_matrix: tensor of shape (N, N) with cosine similarities
    """
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Calculate cosine similarity
    cosine_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    
    return cosine_matrix

def fusion_similarity(embeddings, alpha=0.6):
    """
    Calculate fusion similarity as weighted combination of Pearson and cosine similarity.
    
    Args:
        embeddings: tensor of shape (N, D)
        alpha: weight for combining similarities (0.6 as per friend's optimal setting)
        
    Returns:
        fusion_matrix: tensor of shape (N, N) with fusion similarities
    """
    # Calculate both similarity matrices
    pearson_sim = pearson_similarity(embeddings)
    cosine_sim = cosine_similarity_torch(embeddings)
    
    # Normalize Pearson to [0, 1] range for combination
    pearson_normalized = (pearson_sim + 1.0) / 2.0
    
    # Weighted combination
    fusion_matrix = alpha * pearson_normalized + (1 - alpha) * cosine_sim
    
    return fusion_matrix

def build_fusion_knn_graph(embeddings, k, alpha=0.6):
    """
    Build k-NN graph using fusion similarity.
    
    Args:
        embeddings: tensor of shape (N, D)
        k: number of nearest neighbors
        alpha: fusion weight parameter
        
    Returns:
        indices: edge indices for the graph
        normalized_adj: normalized adjacency matrix
    """
    # Calculate fusion similarity
    sim_matrix = fusion_similarity(embeddings, alpha=alpha)
    
    # Get top-k neighbors
    _, knn_ind = torch.topk(sim_matrix, k, dim=-1)
    adj_size = sim_matrix.size()
    
    # Construct sparse adjacency matrix
    indices0 = torch.arange(knn_ind.shape[0]).to(embeddings.device)
    indices0 = torch.unsqueeze(indices0, 1)
    indices0 = indices0.expand(-1, k)
    indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
    
    return indices, compute_normalized_laplacian(indices, adj_size, embeddings.device)

def compute_normalized_laplacian(indices, adj_size, device):
    """
    Compute normalized Laplacian matrix for the graph.
    """
    adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size).to(device)
    row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
    r_inv_sqrt = torch.pow(row_sum, -0.5)
    rows_inv_sqrt = r_inv_sqrt[indices[0]]
    cols_inv_sqrt = r_inv_sqrt[indices[1]]
    values = rows_inv_sqrt * cols_inv_sqrt
    return torch.sparse.FloatTensor(indices, values, adj_size).to(device)
