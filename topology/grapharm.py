"""
GraphARM: Autoregressive Diffusion Model for Graph Generation

This module implements the GraphARM approach for B-rep topology generation.
The key idea is to use autoregressive diffusion on graphs:
1. Diffusion Ordering Network - learns the optimal ordering for node generation
2. Denoising Network - predicts masked nodes and their connections using GNN

The forward process "absorbs" nodes one-by-one (masking them), and the
reverse process generates nodes in the learned order.
"""

import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import List, Tuple, Optional
from einops import rearrange, repeat


class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention Layer for processing graph structures.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        
        self.gat = GATConv(
            in_dim, 
            out_dim, 
            heads=num_heads, 
            dropout=dropout, 
            concat=concat
        )
        
        if concat:
            self.norm = nn.LayerNorm(out_dim * num_heads)
        else:
            self.norm = nn.LayerNorm(out_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
        Returns:
            Updated node features
        """
        out = self.gat(x, edge_index)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class DiffusionOrderingNetwork(nn.Module):
    """
    Diffusion Ordering Network (DON) for learning the optimal node generation order.
    
    This network learns to predict which nodes should be masked first during 
    the forward diffusion process, effectively learning a data-driven ordering
    for graph generation.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gnn_layers.append(
                GraphAttentionLayer(in_dim, hidden_dim, num_heads, dropout, concat=True)
            )
        
        # Output layer for ordering scores
        self.score_out = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Time embedding for diffusion step
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def sincos_embedding(self, t, dim):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, x, edge_index, edge_attr, mask, t):
        """
        Compute ordering scores for nodes.
        
        Args:
            x: Node features [batch_size, num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            mask: Boolean mask for valid (non-masked) nodes [batch_size, num_nodes]
            t: Diffusion timestep [batch_size]
        
        Returns:
            ordering_scores: Scores indicating order to absorb nodes [batch_size, num_nodes]
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # Time embedding
        t_emb = self.sincos_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [batch_size, hidden_dim]
        
        # Node embedding
        h = self.node_embed(x)  # [batch_size, num_nodes, hidden_dim]
        h = h + t_emb.unsqueeze(1)  # Add time embedding
        
        # Process each graph in batch
        scores_list = []
        for b in range(batch_size):
            h_b = h[b]  # [num_nodes, hidden_dim]
            mask_b = mask[b]  # [num_nodes]
            
            # Get valid nodes
            valid_idx = mask_b.nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                scores_list.append(torch.zeros(num_nodes, device=device))
                continue
            
            h_valid = h_b[valid_idx]  # [num_valid, hidden_dim]
            
            # Create edge index for valid nodes (fully connected)
            n_valid = len(valid_idx)
            row = torch.arange(n_valid, device=device).repeat_interleave(n_valid)
            col = torch.arange(n_valid, device=device).repeat(n_valid)
            # Remove self-loops
            non_self = row != col
            edge_idx = torch.stack([row[non_self], col[non_self]], dim=0)
            
            # Apply GNN layers
            for gnn in self.gnn_layers:
                h_valid = gnn(h_valid, edge_idx)
            
            # Compute scores
            score_valid = self.score_out(h_valid).squeeze(-1)  # [num_valid]
            
            # Map back to full node set
            scores = torch.full((num_nodes,), float('-inf'), device=device)
            scores[valid_idx] = score_valid
            scores_list.append(scores)
        
        return torch.stack(scores_list)  # [batch_size, num_nodes]


class DenoisingNetwork(nn.Module):
    """
    Denoising Network for predicting masked nodes and their connections.
    
    This network takes partially masked graphs and predicts:
    1. Node types/features for masked nodes
    2. Edge connections between masked and existing nodes
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=6, 
                 num_heads=8, num_node_types=1, num_edge_types=5, dropout=0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        
        # Special tokens
        self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Edge type embedding
        self.edge_type_embed = nn.Embedding(num_edge_types + 1, hidden_dim)  # +1 for no-edge
        
        # Position embedding for faces
        self.pos_embed = nn.Embedding(64, hidden_dim)  # Max 64 faces
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output heads
        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_types)
        )
        
        self.edge_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_edge_types + 1)  # +1 for no-edge
        )
    
    def sincos_embedding(self, t, dim):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, x, adj, node_mask, masked_nodes, t):
        """
        Predict node types and edge connections for masked nodes.
        
        Args:
            x: Node features [batch_size, num_nodes, node_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
            node_mask: Valid node mask [batch_size, num_nodes]
            masked_nodes: Boolean mask for nodes to predict [batch_size, num_nodes]
            t: Diffusion timestep [batch_size]
        
        Returns:
            node_logits: Predicted node types [batch_size, num_nodes, num_node_types]
            edge_logits: Predicted edge types [batch_size, num_nodes, num_nodes, num_edge_types+1]
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # Time embedding
        t_emb = self.sincos_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [batch_size, hidden_dim]
        
        # Node embedding
        h = self.node_embed(x)  # [batch_size, num_nodes, hidden_dim]
        
        # Replace masked nodes with mask token
        mask_expanded = masked_nodes.unsqueeze(-1).expand_as(h)
        h = torch.where(mask_expanded, self.mask_token.unsqueeze(0).unsqueeze(0).expand_as(h), h)
        
        # Add position embedding
        pos_ids = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_ids = torch.clamp(pos_ids, 0, 63)  # Clamp to max embedding size
        h = h + self.pos_embed(pos_ids)
        
        # Add time embedding
        h = h + t_emb.unsqueeze(1)
        
        # Create attention mask (can attend to all valid nodes)
        attn_mask = ~node_mask  # True means can't attend
        
        # Apply transformer
        h = self.transformer(h, src_key_padding_mask=attn_mask)  # [batch_size, num_nodes, hidden_dim]
        
        # Node predictions
        node_logits = self.node_out(h)  # [batch_size, num_nodes, num_node_types]
        
        # Edge predictions (for all pairs)
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch_size, num_nodes, num_nodes, hidden_dim]
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch_size, num_nodes, num_nodes, hidden_dim]
        h_pair = torch.cat([h_i, h_j], dim=-1)  # [batch_size, num_nodes, num_nodes, hidden_dim*2]
        edge_logits = self.edge_out(h_pair)  # [batch_size, num_nodes, num_nodes, num_edge_types+1]
        
        # Make edge predictions symmetric
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        
        return node_logits, edge_logits


class GraphARMTopologyModel(nn.Module):
    """
    Complete GraphARM model for B-rep topology generation.
    
    Combines:
    1. Diffusion Ordering Network - learns generation order
    2. Denoising Network - predicts nodes and edges
    
    The model generates face-edge-face adjacency (topology) in an autoregressive manner.
    """
    def __init__(self, max_faces=32, max_edges=30, edge_classes=5, hidden_dim=256, 
                 num_layers=6, num_heads=8, dropout=0.1, use_cf=False, use_pc=False):
        super().__init__()
        
        self.max_faces = max_faces
        self.max_edges = max_edges
        self.edge_classes = edge_classes
        self.hidden_dim = hidden_dim
        self.use_cf = use_cf
        self.use_pc = use_pc
        
        # Node dimension: one-hot face ID + edge count feature
        node_dim = 16
        edge_dim = edge_classes + 1
        
        # Face feature embedding
        self.face_embed = nn.Embedding(max_faces + 1, node_dim)  # +1 for mask
        
        # Ordering network
        self.ordering_net = DiffusionOrderingNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=4,
            num_heads=4,
            dropout=dropout
        )
        
        # Denoising network
        self.denoising_net = DenoisingNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_node_types=1,  # Binary: face exists or not
            num_edge_types=edge_classes,
            dropout=dropout
        )
        
        # Conditional embeddings
        if use_cf:
            self.class_embed = nn.Embedding(11, hidden_dim)  # 10 classes + uncond
        
        if use_pc:
            from model import PointNet2SSG
            # PointNet will be moved to appropriate device during forward pass
            self.pointModel = PointNet2SSG(in_dim=3, out_dim=hidden_dim)
        
        # Cache for sampling
        self.cache = {}
    
    def get_face_features(self, num_faces, batch_size, device):
        """Generate face features as input to the model."""
        # Simple face ID embeddings
        face_ids = torch.arange(num_faces, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.face_embed(face_ids)
    
    def forward_diffusion(self, adj, node_mask, t):
        """
        Forward diffusion: mask nodes according to learned ordering.
        
        Args:
            adj: Face adjacency matrix [batch_size, num_faces, num_faces]
            node_mask: Valid face mask [batch_size, num_faces]
            t: Number of nodes to mask [batch_size]
        
        Returns:
            masked_adj: Adjacency with masked nodes
            masked_nodes: Boolean mask indicating which nodes are masked
            ordering: The learned ordering of nodes
        """
        batch_size, num_faces, _ = adj.shape
        device = adj.device
        
        # Get face features
        x = self.get_face_features(num_faces, batch_size, device)
        
        # Compute ordering scores
        ordering_scores = self.ordering_net(x, None, None, node_mask, t)
        
        # For each sample, mask t nodes with highest scores
        masked_nodes = torch.zeros_like(node_mask)
        for b in range(batch_size):
            valid_idx = node_mask[b].nonzero(as_tuple=True)[0]
            scores = ordering_scores[b, valid_idx]
            num_to_mask = min(t[b].item(), len(valid_idx))
            if num_to_mask > 0:
                _, top_idx = torch.topk(scores, num_to_mask)
                masked_nodes[b, valid_idx[top_idx]] = True
        
        # Create masked adjacency
        masked_adj = adj.clone()
        for b in range(batch_size):
            mask = masked_nodes[b]
            masked_adj[b, mask, :] = 0
            masked_adj[b, :, mask] = 0
        
        return masked_adj, masked_nodes, ordering_scores
    
    def forward(self, fef_adj, node_mask, class_label=None, point_data=None):
        """
        Training forward pass.
        
        Args:
            fef_adj: Face-edge-face adjacency [batch_size, num_faces, num_faces]
            node_mask: Valid face mask [batch_size, num_faces]
            class_label: Optional class condition [batch_size, 1]
            point_data: Optional point cloud condition [batch_size, 3, 2000]
        
        Returns:
            Loss dictionary
        """
        batch_size, num_faces, _ = fef_adj.shape
        device = fef_adj.device
        
        # Random timestep (number of nodes to mask)
        num_valid = node_mask.sum(dim=1)
        t = torch.randint(1, num_faces, (batch_size,), device=device)
        t = torch.min(t, num_valid)
        
        # Forward diffusion
        masked_adj, masked_nodes, ordering_scores = self.forward_diffusion(fef_adj, node_mask, t)
        
        # Get face features
        x = self.get_face_features(num_faces, batch_size, device)
        
        # Denoising prediction
        node_logits, edge_logits = self.denoising_net(x, masked_adj, node_mask, masked_nodes, t)
        
        # Compute losses
        # Node loss (predict which positions have faces)
        node_target = node_mask.float().unsqueeze(-1)
        node_loss = F.binary_cross_entropy_with_logits(
            node_logits[masked_nodes].view(-1),
            node_target[masked_nodes].view(-1)
        )
        
        # Edge loss (predict adjacency for masked nodes)
        edge_target = fef_adj.long()
        edge_loss_mask = masked_nodes.unsqueeze(-1) | masked_nodes.unsqueeze(-2)
        edge_loss_mask = edge_loss_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        
        if edge_loss_mask.sum() > 0:
            edge_loss = F.cross_entropy(
                edge_logits[edge_loss_mask].view(-1, self.edge_classes + 1),
                edge_target[edge_loss_mask].view(-1),
                reduction='mean'
            )
        else:
            edge_loss = torch.tensor(0.0, device=device)
        
        # Ordering loss (encourage ordering that makes generation easier)
        # We want high scores for peripheral nodes
        degree = fef_adj.sum(dim=-1).float()  # [batch_size, num_faces]
        ordering_target = -degree  # Negative degree = peripheral nodes first
        ordering_loss = F.mse_loss(
            ordering_scores[node_mask],
            ordering_target[node_mask]
        )
        
        total_loss = node_loss + edge_loss + 0.1 * ordering_loss
        
        return {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss,
            'ordering_loss': ordering_loss
        }
    
    @torch.no_grad()
    def sample(self, num_samples, num_faces=None, class_label=None, point_data=None):
        """
        Generate face-edge-face adjacency matrices.
        
        Args:
            num_samples: Number of samples to generate
            num_faces: Number of faces per sample (if None, randomly chosen)
            class_label: Optional class condition [num_samples, 1]
            point_data: Optional point cloud condition [num_samples, 3, 2000]
        
        Returns:
            fef_adj: Generated adjacency matrices [num_samples, max_faces, max_faces]
        """
        device = next(self.parameters()).device
        
        # Initialize with empty graphs
        if num_faces is None:
            num_faces = torch.randint(4, self.max_faces + 1, (num_samples,), device=device)
        elif isinstance(num_faces, int):
            num_faces = torch.full((num_samples,), num_faces, device=device)
        
        # Start with all nodes masked
        adj = torch.zeros(num_samples, self.max_faces, self.max_faces, device=device)
        node_mask = torch.zeros(num_samples, self.max_faces, dtype=torch.bool, device=device)
        
        # Set valid node positions
        for b in range(num_samples):
            node_mask[b, :num_faces[b]] = True
        
        # Currently all are masked
        masked_nodes = node_mask.clone()
        
        # Reverse diffusion: generate nodes one by one
        for step in range(self.max_faces):
            # Get features
            x = self.get_face_features(self.max_faces, num_samples, device)
            
            t = (masked_nodes.sum(dim=1)).float()  # Remaining masked nodes
            
            if t.max() == 0:
                break
            
            # Compute ordering for which node to generate next
            ordering_scores = self.ordering_net(x, None, None, masked_nodes, t.long())
            
            # Select node with lowest score (generate in reverse order)
            ordering_scores = ordering_scores.masked_fill(~masked_nodes, float('inf'))
            next_node = ordering_scores.argmin(dim=1)  # [num_samples]
            
            # Predict edges for the selected node
            current_masked = torch.zeros_like(masked_nodes)
            for b in range(num_samples):
                if masked_nodes[b, next_node[b]]:
                    current_masked[b, next_node[b]] = True
            
            _, edge_logits = self.denoising_net(x, adj, node_mask, current_masked, t.long())
            
            # Sample edges
            for b in range(num_samples):
                node_idx = next_node[b].item()
                if not masked_nodes[b, node_idx]:
                    continue
                
                # Get edge predictions for this node to all non-masked nodes
                for j in range(self.max_faces):
                    if j != node_idx and node_mask[b, j] and not masked_nodes[b, j]:
                        logits = edge_logits[b, node_idx, j]
                        edge_type = torch.distributions.Categorical(logits=logits).sample()
                        adj[b, node_idx, j] = edge_type
                        adj[b, j, node_idx] = edge_type  # Symmetric
                
                # Unmask the node
                masked_nodes[b, node_idx] = False
        
        return adj.long()


class GraphARMDataset(torch.utils.data.Dataset):
    """
    Dataset for training GraphARM topology model.
    """
    def __init__(self, path, args):
        from utils import load_data_with_prefix, check_step_ok
        import pickle
        
        self.data = load_data_with_prefix(path, '.pkl')
        self.max_face = args.max_face
        self.edge_classes = args.edge_classes
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc
        
        # Text to int mapping for class labels
        self.text2int = {
            'bathtub': 0, 'bed': 1, 'bench': 2, 'bookshelf': 3, 'cabinet': 4,
            'chair': 5, 'couch': 6, 'lamp': 7, 'sofa': 8, 'table': 9
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        from utils import pad_zero
        
        with open(self.data[idx], "rb") as tf:
            data = pickle.load(tf)
        
        fef_adj = data['fef_adj']  # [nf, nf]
        nf = fef_adj.shape[0]
        
        # Pad to max_face
        fef_adj, mask = pad_zero(fef_adj, max_len=self.max_face, dim=1)
        
        result = [
            torch.from_numpy(fef_adj).long(),  # [max_face, max_face]
            torch.from_numpy(mask)              # [max_face]
        ]
        
        if self.use_cf:
            data_class = self.text2int[data['name'].split('_')[0]] + 1
            result.append(torch.LongTensor([data_class]))
        
        if self.use_pc and 'pc' in data:
            result.append(torch.from_numpy(data['pc']))
        
        return tuple(result)
