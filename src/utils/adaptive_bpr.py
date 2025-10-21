"""
Adaptive BPR weighting based on user activity levels.
Improves representation quality for low-frequency users as per friend's reproduction.
"""
import torch
import numpy as np

class AdaptiveBPRLoss(torch.nn.Module):
    """
    Adaptive BPR Loss with user activity-based weighting.
    
    This implements user activity-based adaptive weights in the BPR loss function
    to improve long-tail item coverage and low-frequency user representation quality.
    """
    
    def __init__(self, dataset, gamma=0.5):
        """
        Args:
            dataset: training dataset to calculate user activity levels
            gamma: weighting parameter for adaptive scaling (0.5 as per friend's setting)
        """
        super(AdaptiveBPRLoss, self).__init__()
        self.gamma = gamma
        self.user_activity_weights = self._calculate_user_activity_weights(dataset)
        
    def _calculate_user_activity_weights(self, dataset):
        """
        Calculate user activity weights based on interaction frequency.
        
        Args:
            dataset: RecDataset with interaction data
            
        Returns:
            weights: tensor of weights for each user
        """
        # Count interactions per user
        user_interactions = {}
        for idx in range(len(dataset)):
            row = dataset[idx]
            user_id = row[dataset.uid_field]
            user_interactions[user_id] = user_interactions.get(user_id, 0) + 1
        
        # Convert to numpy array
        n_users = dataset.user_num
        interaction_counts = np.zeros(n_users)
        
        for user_id, count in user_interactions.items():
            if user_id < n_users:  # Safety check
                interaction_counts[user_id] = count
        
        # Calculate adaptive weights
        # Higher weights for users with fewer interactions (long-tail users)
        max_interactions = np.max(interaction_counts)
        min_interactions = np.max([np.min(interaction_counts[interaction_counts > 0]), 1])  # Avoid zero
        
        # Normalize interaction counts to [0, 1]
        normalized_counts = (interaction_counts - min_interactions) / (max_interactions - min_interactions + 1e-8)
        
        # Adaptive weight: w_u = 1 + γ * (1 - normalized_activity)
        # This gives higher weights to less active users
        weights = 1.0 + self.gamma * (1.0 - normalized_counts)
        
        print(f"User activity weights - Min: {weights.min():.3f}, Max: {weights.max():.3f}, Mean: {weights.mean():.3f}")
        
        return torch.FloatTensor(weights)
    
    def forward(self, pos_scores, neg_scores, users):
        """
        Calculate adaptive BPR loss.
        
        Args:
            pos_scores: positive item scores
            neg_scores: negative item scores
            users: user indices for the batch
            
        Returns:
            adaptive_bpr_loss: weighted BPR loss
        """
        # Get user weights for this batch
        batch_weights = self.user_activity_weights[users].to(pos_scores.device)
        
        # Standard BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        
        # Apply adaptive weights
        weighted_loss = batch_weights * bpr_loss
        
        return weighted_loss.mean()

def create_adaptive_bpr_loss(dataset, gamma=0.5):
    """
    Factory function to create adaptive BPR loss.
    
    Args:
        dataset: training dataset
        gamma: adaptive weighting parameter
        
    Returns:
        AdaptiveBPRLoss instance
    """
    return AdaptiveBPRLoss(dataset, gamma=gamma)
