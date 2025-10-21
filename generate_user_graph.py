#!/usr/bin/env python3
"""
Generate user-user graph based on item interactions
This replaces the missing tools/generate-u-u-matrix.py
"""

import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_interactions(dataset_path):
    """Load user-item interactions from .inter file"""
    inter_file = None
    for file in os.listdir(dataset_path):
        if file.endswith('.inter'):
            inter_file = os.path.join(dataset_path, file)
            break
    
    if not inter_file:
        raise FileNotFoundError(f"No .inter file found in {dataset_path}")
    
    # Read interaction file (format: userID\titemID\trating\ttimestamp\tx_label)
    with open(inter_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    interactions = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            user_id = int(parts[0])
            item_id = int(parts[1])
            interactions.append((user_id, item_id))
    
    return interactions

def create_user_item_matrix(interactions, num_users, num_items):
    """Create user-item interaction matrix"""
    users, items = zip(*interactions)
    data = np.ones(len(interactions))
    
    # Create sparse matrix
    user_item_matrix = csr_matrix((data, (users, items)), shape=(num_users, num_items))
    return user_item_matrix

def generate_user_graph(dataset_path, k=10, similarity_threshold=0.1):
    """
    Generate user-user graph based on item interaction similarity
    
    Args:
        dataset_path: Path to dataset directory
        k: Maximum number of connections per user
        similarity_threshold: Minimum similarity to create connection
    """
    print(f"Generating user graph for dataset: {dataset_path}")
    
    # Load user and item mappings
    u_mapping = pd.read_csv(os.path.join(dataset_path, 'u_id_mapping.csv'), header=None)
    i_mapping = pd.read_csv(os.path.join(dataset_path, 'i_id_mapping.csv'), header=None)
    
    num_users = len(u_mapping)
    num_items = len(i_mapping)
    
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    
    # Load interactions
    interactions = load_interactions(dataset_path)
    print(f"Number of interactions: {len(interactions)}")
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(interactions, num_users, num_items)
    
    # Calculate user-user similarity (cosine similarity)
    print("Calculating user similarities...")
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Generate user graph dictionary
    user_graph_dict = {}
    
    print("Building user graph...")
    for user_id in range(num_users):
        # Get similarities for this user
        similarities = user_similarity[user_id]
        
        # Get indices of most similar users (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        similar_indices = similar_indices[similar_indices != user_id]  # Remove self
        
        # Filter by similarity threshold and take top k
        valid_users = []
        valid_similarities = []
        
        for idx in similar_indices:
            if similarities[idx] > similarity_threshold and len(valid_users) < k:
                valid_users.append(int(idx))
                valid_similarities.append(float(similarities[idx]))
        
        # If no similar users found, connect to random users
        if len(valid_users) == 0:
            # Connect to a few random users with small weights
            num_random = min(3, num_users - 1)
            possible_users = list(range(num_users))
            possible_users.remove(user_id)
            
            if len(possible_users) > 0:
                random_users = np.random.choice(possible_users, size=num_random, replace=False)
                valid_users = random_users.tolist()
                valid_similarities = [0.01] * len(valid_users)  # Small similarity
        
        user_graph_dict[user_id] = (valid_users, valid_similarities)
        
        if user_id % 1000 == 0:
            print(f"Processed {user_id}/{num_users} users")
    
    return user_graph_dict

def main():
    parser = argparse.ArgumentParser(description='Generate user-user graph')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (clothing, sports, baby)')
    parser.add_argument('--k', type=int, default=10, help='Max connections per user')
    parser.add_argument('--threshold', type=float, default=0.1, help='Similarity threshold')
    
    args = parser.parse_args()
    
    dataset_path = f'data/{args.dataset}'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return
    
    # Generate user graph
    user_graph_dict = generate_user_graph(dataset_path, k=args.k, similarity_threshold=args.threshold)
    
    # Save user graph
    output_file = os.path.join(dataset_path, 'user_graph_dict.npy')
    np.save(output_file, user_graph_dict)
    
    print(f"User graph saved to {output_file}")
    print(f"Generated graph for {len(user_graph_dict)} users")
    
    # Print some statistics
    connection_counts = [len(connections[0]) for connections in user_graph_dict.values()]
    avg_connections = np.mean(connection_counts)
    print(f"Average connections per user: {avg_connections:.2f}")
    print(f"Min connections: {min(connection_counts)}")
    print(f"Max connections: {max(connection_counts)}")

if __name__ == "__main__":
    main()
