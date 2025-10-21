# coding: utf-8
# Simplified SEA model matching the paper exactly
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender

class CLUBSample(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, device='cpu'):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
        self.to(device)

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        negative = (-(mu - y_samples[random_index]) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        bound = (positive - negative).mean()
        return torch.clamp(bound / 2., min=0.0)

class SEA(GeneralRecommender):
    def __init__(self, config, dataset, args):
        super(SEA, self).__init__(config, dataset)

        # Basic configuration
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        
        # SEA-specific parameters
        self.alpha_contrast = args.alpha_contrast
        self.temp = args.temp
        self.beta = args.beta
        
        # Create multimodal adjacency matrix for item-item graph
        self.mm_adj = self._create_mm_adjacency()
        
        # Create user-item bipartite graph
        self.edge_index = self._create_bipartite_graph(dataset)
        
        # Initialize GCN layers for different modalities
        if self.v_feat is not None:
            self.v_gcn = GCN(self.n_users, self.n_items, self.embedding_size, 
                           self.feat_embed_dim, self.v_feat, self.device)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.n_users, self.n_items, self.embedding_size, 
                           self.feat_embed_dim, self.t_feat, self.device)
        
        # User attention weights for combining modalities
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_users, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)
        
        # Final result embeddings
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_users + self.n_items, self.embedding_size))))
        
    def _create_mm_adjacency(self):
        """Create multimodal adjacency matrix for item-item graph"""
        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))
        
        if os.path.exists(mm_adj_file):
            mm_adj = torch.load(mm_adj_file)
        else:
            # Create adjacency based on multimodal features
            if self.v_feat is not None:
                image_adj = self._get_knn_adj_mat(self.v_feat)
                mm_adj = image_adj
            if self.t_feat is not None:
                text_adj = self._get_knn_adj_mat(self.t_feat)
                mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            torch.save(mm_adj, mm_adj_file)
        
        return mm_adj.to(self.device)
    
    def _get_knn_adj_mat(self, features):
        """Create k-nearest neighbor adjacency matrix"""
        context_norm = F.normalize(features, p=2, dim=-1)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        
        # Construct sparse adjacency matrix
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        # Normalize
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def _create_bipartite_graph(self, dataset):
        """Create bipartite user-item graph"""
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        rows = train_interactions.row
        cols = train_interactions.col + self.n_users
        edge_index = np.column_stack((rows, cols))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        return torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
    
    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        
        # Get representations from different modalities
        if self.v_feat is not None:
            v_rep = self.v_gcn(self.edge_index)
        if self.t_feat is not None:
            t_rep = self.t_gcn(self.edge_index)
        
        # Combine user representations from different modalities
        user_v_rep = v_rep[:self.n_users].unsqueeze(2)
        user_t_rep = t_rep[:self.n_users].unsqueeze(2)
        user_rep = torch.cat((user_v_rep, user_t_rep), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)
        
        # Get item representations
        item_v_rep = v_rep[self.n_users:]
        item_t_rep = t_rep[self.n_users:]
        
        # Apply item-item multimodal graph convolution
        item_rep = torch.cat((item_v_rep, item_t_rep), dim=1)
        for _ in range(self.n_layers):
            item_rep = torch.sparse.mm(self.mm_adj, item_rep)
        
        # Update result embeddings
        self.result_embed.data = torch.cat((user_rep, item_rep), dim=0)
        
        # Calculate scores
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        
        # Split representations for mutual information computation
        # Split each modality into two parts (generic and unique)
        v_dim = item_v_rep.shape[1] // 2
        t_dim = item_t_rep.shape[1] // 2
        
        v_generic = item_v_rep[:, :v_dim]
        v_unique = item_v_rep[:, v_dim:]
        t_generic = item_t_rep[:, :t_dim]
        t_unique = item_t_rep[:, t_dim:]
        
        return pos_scores, neg_scores, v_generic, v_unique, t_generic, t_unique
    
    def calculate_loss(self, interaction, club):
        pos_scores, neg_scores, v_generic, v_unique, t_generic, t_unique = self.forward(interaction)
        
        # BPR loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Alignment loss: maximize MI between generic parts of different modalities
        align_loss = self.solosimloss(v_generic, t_generic, self.temp)
        
        # Distancing loss: minimize MI between generic and unique parts within same modality
        v_distance_loss = club.forward(v_generic, v_unique)
        t_distance_loss = club.forward(t_generic, t_unique)
        
        # Combined loss according to paper's Equation 19
        total_loss = bpr_loss + self.alpha_contrast * align_loss + self.beta * (v_distance_loss + t_distance_loss)
        
        return total_loss, (v_generic, v_unique, t_generic, t_unique)
    
    def solosimloss(self, view1, view2, temperature):
        """Contrastive loss for alignment (Equation 16 in paper)"""
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
    
    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class GCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_size, feat_embed_dim, features, device):
        super(GCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.feat_embed_dim = feat_embed_dim
        self.device = device
        
        # User preference embeddings
        self.user_preference = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(n_users, feat_embed_dim), dtype=torch.float32, requires_grad=True)))
        
        # Feature transformation
        self.feat_transform = nn.Linear(features.shape[1], feat_embed_dim)
        
        # GCN layer
        self.gcn_layer = GraphConv()
        
        self.to(device)
    
    def forward(self, edge_index):
        # Transform features
        item_features = self.feat_transform(self.features)
        
        # Combine user preferences and item features
        x = torch.cat((self.user_preference, item_features), dim=0)
        x = F.normalize(x, dim=1)
        
        # Apply GCN layers
        h1 = self.gcn_layer(x, edge_index)
        h2 = self.gcn_layer(h1, edge_index)
        
        return x + h1 + h2

class GraphConv(MessagePassing):
    def __init__(self):
        super(GraphConv, self).__init__(aggr='add')
    
    def forward(self, x, edge_index):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j, edge_index):
        row, col = edge_index
        deg = degree(row, x_j.size(0), dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j
