"""
GNN-based Topology Generation Module for DTGBrepGen

This module implements a Graph Neural Network (GNN) based approach for generating
B-rep topology. Instead of using a Transformer-based VAE for generating face-edge 
adjacency matrices, we use Graph Attention Networks (GAT) to directly generate 
the topology structure.

The GNN model predicts the face-edge-face adjacency matrix through a two-stage process:
1. Generate node (face) features using a graph encoder
2. Predict edge weights between faces using edge predictors

This approach leverages the natural graph structure of B-rep models where:
- Nodes represent faces
- Edges represent connections between faces (sharing edges)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, GCNConv, GraphConv, MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat
import numpy as np

try:
    from model import PointNet2SSG, Embedder
except ImportError:
    pass


class GraphAttentionLayer(nn.Module):
    """
    Custom Graph Attention Layer with multi-head attention.
    """
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat
        
        if concat:
            assert out_features % n_heads == 0
            self.head_dim = out_features // n_heads
        else:
            self.head_dim = out_features
            
        # Learnable weight matrices
        self.W = nn.Linear(in_features, n_heads * self.head_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj_mask=None):
        """
        Args:
            h: Node features [batch, num_nodes, in_features]
            adj_mask: Adjacency mask [batch, num_nodes, num_nodes]
        Returns:
            Updated node features [batch, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        Wh = self.W(h)  # [batch, num_nodes, n_heads * head_dim]
        Wh = Wh.view(batch_size, num_nodes, self.n_heads, self.head_dim)  # [batch, N, heads, head_dim]
        Wh = Wh.permute(0, 2, 1, 3)  # [batch, heads, N, head_dim]
        
        # Compute attention scores
        Wh_i = Wh.unsqueeze(3)  # [batch, heads, N, 1, head_dim]
        Wh_j = Wh.unsqueeze(2)  # [batch, heads, 1, N, head_dim]
        
        # Concatenate and compute attention
        a_input = torch.cat([Wh_i.expand(-1, -1, -1, num_nodes, -1),
                            Wh_j.expand(-1, -1, num_nodes, -1, -1)], dim=-1)  # [batch, heads, N, N, 2*head_dim]
        
        e = self.leaky_relu(torch.einsum('bhijd,hd->bhij', a_input, self.a))  # [batch, heads, N, N]
        
        # Apply mask if provided
        if adj_mask is not None:
            adj_mask = adj_mask.unsqueeze(1)  # [batch, 1, N, N]
            e = e.masked_fill(~adj_mask, float('-inf'))
        
        # Softmax and dropout
        attention = F.softmax(e, dim=-1)
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
        attention = self.dropout(attention)
        
        # Apply attention to values
        h_prime = torch.matmul(attention, Wh)  # [batch, heads, N, head_dim]
        
        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous()  # [batch, N, heads, head_dim]
            h_prime = h_prime.view(batch_size, num_nodes, -1)  # [batch, N, out_features]
        else:
            h_prime = h_prime.mean(dim=1)  # [batch, N, head_dim]
            
        return h_prime


class GNNEncoder(nn.Module):
    """
    GNN Encoder for processing face graph structure.
    Uses multiple layers of Graph Attention to learn rich face representations.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(n_layers):
            self.layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_mask=None):
        """
        Args:
            x: Node features [batch, num_nodes, in_dim]
            adj_mask: Fully connected mask [batch, num_nodes, num_nodes]
        Returns:
            Node embeddings [batch, num_nodes, out_dim]
        """
        x = self.input_proj(x)
        x = F.gelu(x)
        
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = x
            x = layer(x, adj_mask)
            x = self.dropout(x)
            x = norm(x + residual)
            x = F.gelu(x)
        
        x = self.output_proj(x)
        return x


class EdgePredictor(nn.Module):
    """
    Predicts edge weights/counts between nodes (faces).
    Uses node embeddings to predict the number of edges between each pair of faces.
    """
    def __init__(self, node_dim, hidden_dim, num_edge_classes):
        super().__init__()
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_edge_classes)
        )
        
    def forward(self, node_embeddings, node_mask=None):
        """
        Args:
            node_embeddings: [batch, num_nodes, node_dim]
            node_mask: [batch, num_nodes]
        Returns:
            edge_logits: [batch, num_nodes, num_nodes, num_edge_classes]
        """
        batch_size, num_nodes, node_dim = node_embeddings.shape
        
        # Create pairwise node features
        nodes_i = node_embeddings.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, D]
        nodes_j = node_embeddings.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, D]
        
        edge_features = torch.cat([nodes_i, nodes_j], dim=-1)  # [B, N, N, 2D]
        
        edge_logits = self.edge_mlp(edge_features)  # [B, N, N, num_classes]
        
        # Make symmetric by averaging upper and lower triangular
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        
        # Apply mask
        if node_mask is not None:
            mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)  # [B, N, N]
            mask = mask.unsqueeze(-1)  # [B, N, N, 1]
            edge_logits = edge_logits * mask
        
        # Zero out diagonal (no self-loops)
        diag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=node_embeddings.device).unsqueeze(0).unsqueeze(-1)
        edge_logits = edge_logits * diag_mask
        
        return edge_logits


class GNNTopologyGenerator(nn.Module):
    """
    GNN-based Topology Generator for B-rep models.
    
    This model uses Graph Neural Networks to generate the face-edge-face adjacency matrix,
    which defines the topological structure of a B-rep model.
    
    Architecture:
    1. Face embedding layer: Embeds each face with learnable positional encodings
    2. GNN Encoder: Multiple layers of Graph Attention for learning face relationships
    3. Edge Predictor: MLP that predicts edge counts between face pairs
    
    Training: Uses cross-entropy loss to predict the number of edges between face pairs.
    Generation: Samples from the predicted edge distributions to create adjacency matrix.
    """
    
    def __init__(self, 
                 max_faces=32, 
                 d_model=128, 
                 n_layers=4, 
                 n_heads=4,
                 num_edge_classes=5, 
                 dropout=0.1,
                 use_cf=False,
                 use_pc=False):
        super().__init__()
        
        self.max_faces = max_faces
        self.d_model = d_model
        self.num_edge_classes = num_edge_classes
        self.use_cf = use_cf
        self.use_pc = use_pc
        
        # Face positional embeddings
        self.face_embedding = nn.Embedding(max_faces, d_model)
        
        # Optional class conditioning
        if use_cf:
            self.class_embed = nn.Embedding(12, d_model)  # 11 classes + uncond
        
        # Optional point cloud conditioning
        if use_pc:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.pointModel = PointNet2SSG(in_dim=3, out_dim=d_model).to(device)
        
        # GNN Encoder
        self.gnn_encoder = GNNEncoder(
            in_dim=d_model,
            hidden_dim=d_model,
            out_dim=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Edge Predictor
        self.edge_predictor = EdgePredictor(
            node_dim=d_model,
            hidden_dim=d_model * 2,
            num_edge_classes=num_edge_classes
        )
        
        # VAE components for latent space
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)
        
        # Decoder from latent
        self.latent_to_node = nn.Linear(d_model, d_model)
        
    def encode(self, adj_matrix, node_mask, class_label=None, point_data=None):
        """
        Encode the adjacency matrix into a latent distribution.
        
        Args:
            adj_matrix: Ground truth adjacency matrix [batch, max_faces, max_faces]
            node_mask: Valid node mask [batch, max_faces]
            class_label: Optional class conditioning [batch, 1]
            point_data: Optional point cloud data [batch, 3, num_points]
        Returns:
            mu, logvar: Parameters of the latent distribution
        """
        batch_size = adj_matrix.shape[0]
        device = adj_matrix.device
        
        # Get face embeddings
        face_ids = torch.arange(self.max_faces, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.face_embedding(face_ids)  # [batch, max_faces, d_model]
        
        # Add class conditioning
        if self.use_cf and class_label is not None:
            class_emb = self.class_embed(class_label)  # [batch, 1, d_model]
            x = x + class_emb
        
        # Add point cloud conditioning
        if self.use_pc and point_data is not None:
            point_data = rearrange(point_data, 'b d n -> b n d')
            point_data = point_data.float()
            pc_emb = self.pointModel(point_data)  # [batch, d_model]
            pc_emb = pc_emb.unsqueeze(1)  # [batch, 1, d_model]
            x = x + pc_emb
        
        # Create adjacency mask from adj_matrix for message passing
        adj_mask = (adj_matrix > 0) | torch.eye(self.max_faces, device=device).bool().unsqueeze(0)
        adj_mask = adj_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        
        # Apply GNN encoder
        x = self.gnn_encoder(x, adj_mask)  # [batch, max_faces, d_model]
        
        # Pool to get global representation
        x_masked = x * node_mask.unsqueeze(-1)
        x_pooled = x_masked.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, d_model]
        
        mu = self.fc_mu(x_pooled)
        logvar = self.fc_logvar(x_pooled)
        
        return mu, logvar
    
    def decode(self, z, node_mask, class_label=None, point_data=None):
        """
        Decode latent vector to adjacency matrix logits.
        
        Args:
            z: Latent vector [batch, d_model]
            node_mask: Valid node mask [batch, max_faces]
            class_label: Optional class conditioning [batch, 1]
            point_data: Optional point cloud data [batch, 3, num_points]
        Returns:
            edge_logits: [batch, max_faces, max_faces, num_edge_classes]
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Get face embeddings
        face_ids = torch.arange(self.max_faces, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.face_embedding(face_ids)  # [batch, max_faces, d_model]
        
        # Condition on latent
        z_expand = z.unsqueeze(1).expand(-1, self.max_faces, -1)  # [batch, max_faces, d_model]
        x = x + self.latent_to_node(z_expand)
        
        # Add class conditioning
        if self.use_cf and class_label is not None:
            class_emb = self.class_embed(class_label)  # [batch, 1, d_model]
            x = x + class_emb
        
        # Add point cloud conditioning
        if self.use_pc and point_data is not None:
            point_data = rearrange(point_data, 'b d n -> b n d')
            point_data = point_data.float()
            pc_emb = self.pointModel(point_data)  # [batch, d_model]
            pc_emb = pc_emb.unsqueeze(1)  # [batch, 1, d_model]
            x = x + pc_emb
        
        # Use fully connected graph for generation
        adj_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        
        # Apply GNN encoder
        x = self.gnn_encoder(x, adj_mask)
        
        # Predict edges
        edge_logits = self.edge_predictor(x, node_mask)
        
        return edge_logits
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, adj_matrix, node_mask, class_label=None, point_data=None):
        """
        Forward pass for training.
        
        Args:
            adj_matrix: Ground truth adjacency matrix [batch, max_faces, max_faces]
            node_mask: Valid node mask [batch, max_faces]
            class_label: Optional class conditioning [batch, 1]
            point_data: Optional point cloud data [batch, 3, num_points]
        Returns:
            edge_logits: Predicted edge logits [batch, max_faces, max_faces, num_edge_classes]
            mu, logvar: Latent distribution parameters
        """
        # Encode
        mu, logvar = self.encode(adj_matrix, node_mask, class_label, point_data)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        edge_logits = self.decode(z, node_mask, class_label, point_data)
        
        return edge_logits, mu, logvar
    
    def sample(self, num_samples, class_label=None, point_data=None, num_faces=None, temperature=1.0):
        """
        Sample new topologies from the model.
        
        Args:
            num_samples: Number of samples to generate
            class_label: Optional class conditioning [num_samples, 1]
            point_data: Optional point cloud data [num_samples, 3, num_points]
            num_faces: Number of faces per sample (None for random)
            temperature: Sampling temperature (lower = more deterministic)
        Returns:
            adj_matrices: Generated adjacency matrices [num_samples, max_faces, max_faces]
        """
        device = next(self.parameters()).device
        
        # Sample latent
        z = torch.randn(num_samples, self.d_model, device=device)
        
        # Create node mask
        if num_faces is None:
            # Random number of faces between 4 and max_faces
            num_faces = torch.randint(4, self.max_faces + 1, (num_samples,), device=device)
            node_mask = torch.arange(self.max_faces, device=device).unsqueeze(0) < num_faces.unsqueeze(1)
        elif isinstance(num_faces, int):
            node_mask = torch.ones(num_samples, self.max_faces, dtype=torch.bool, device=device)
            node_mask[:, num_faces:] = False
        else:
            node_mask = torch.arange(self.max_faces, device=device).unsqueeze(0) < num_faces.unsqueeze(1)
        
        # Prepare point data embeddings if needed
        point_embd = None
        if self.use_pc and point_data is not None:
            point_data_proc = rearrange(point_data, 'b d n -> b n d')
            point_data_proc = point_data_proc.float()
            point_embd = self.pointModel(point_data_proc)
            point_embd = point_embd.unsqueeze(1)
        
        # Decode
        with torch.no_grad():
            edge_logits = self.decode(z, node_mask, class_label, point_data)  # [B, N, N, C]
        
        # Sample from logits
        edge_logits = edge_logits / temperature
        edge_probs = F.softmax(edge_logits, dim=-1)  # [B, N, N, C]
        
        # Sample edge counts
        edge_counts = torch.zeros(num_samples, self.max_faces, self.max_faces, 
                                  dtype=torch.long, device=device)
        
        # Only sample upper triangle
        upper_indices = torch.triu_indices(self.max_faces, self.max_faces, offset=1)
        for b in range(num_samples):
            for i, j in zip(upper_indices[0], upper_indices[1]):
                if node_mask[b, i] and node_mask[b, j]:
                    probs = edge_probs[b, i, j]
                    count = torch.multinomial(probs, 1).item()
                    edge_counts[b, i, j] = count
                    edge_counts[b, j, i] = count  # Symmetric
        
        return edge_counts
    
    def compute_loss(self, edge_logits, target_adj, node_mask, mu, logvar, kl_weight=1.0):
        """
        Compute the training loss.
        
        Args:
            edge_logits: Predicted logits [batch, N, N, num_classes]
            target_adj: Target adjacency [batch, N, N]
            node_mask: Node mask [batch, N]
            mu, logvar: Latent distribution parameters
            kl_weight: Weight for KL divergence term
        Returns:
            total_loss, recon_loss, kl_loss
        """
        batch_size = edge_logits.shape[0]
        
        # Create edge mask (upper triangle, valid nodes)
        edge_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        upper_mask = torch.triu(torch.ones(self.max_faces, self.max_faces, 
                                           device=edge_logits.device), diagonal=1).bool()
        edge_mask = edge_mask & upper_mask.unsqueeze(0)
        
        # Compute reconstruction loss
        target = target_adj.long()  # [B, N, N]
        
        # Reshape for cross entropy
        edge_logits_flat = edge_logits[edge_mask]  # [num_valid_edges, num_classes]
        target_flat = target[edge_mask]  # [num_valid_edges]
        
        if edge_logits_flat.numel() > 0:
            recon_loss = F.cross_entropy(edge_logits_flat, target_flat, reduction='mean')
        else:
            recon_loss = torch.tensor(0.0, device=edge_logits.device)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


class GNNTopologyTrainer:
    """
    Trainer class for GNN Topology Generator.
    """
    
    def __init__(self, args, train_dataset, val_dataset):
        self.iters = 0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        self.use_cf = args.use_cf
        self.use_pc = getattr(args, 'use_pc', False)
        
        # Initialize model
        model = GNNTopologyGenerator(
            max_faces=args.max_face,
            d_model=args.GNNTopologyModel.get('d_model', 128),
            n_layers=args.GNNTopologyModel.get('n_layers', 4),
            n_heads=args.GNNTopologyModel.get('n_heads', 4),
            num_edge_classes=args.edge_classes,
            dropout=args.GNNTopologyModel.get('dropout', 0.1),
            use_cf=self.use_cf,
            use_pc=self.use_pc
        )
        
        model = nn.DataParallel(model)
        self.model = model.to(self.device).train()
        
        # Data loaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=8
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_one_epoch(self):
        import wandb
        from tqdm import tqdm
        
        self.model.train()
        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")
        
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                
                if self.use_cf:
                    fef_adj, node_mask, class_label = data
                else:
                    fef_adj, node_mask = data
                    class_label = None
                
                self.optimizer.zero_grad()
                
                # Forward pass
                edge_logits, mu, logvar = self.model(fef_adj, node_mask, class_label)
                
                # Compute loss
                loss, recon_loss, kl_loss = self.model.module.compute_loss(
                    edge_logits, fef_adj, node_mask, mu, logvar
                )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # Logging
            if self.iters % 20 == 0:
                wandb.log({
                    "Loss": loss.item(),
                    "Recon_Loss": recon_loss.item(),
                    "KL_Loss": kl_loss.item()
                }, step=self.iters)
                print(f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
            
            self.iters += 1
            progress_bar.update(1)
        
        progress_bar.close()
        self.epoch += 1
    
    def test_val(self):
        import wandb
        from tqdm import tqdm
        
        self.model.eval()
        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description("Testing")
        total_loss = []
        
        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                
                if self.use_cf:
                    fef_adj, node_mask, class_label = data
                else:
                    fef_adj, node_mask = data
                    class_label = None
                
                edge_logits, mu, logvar = self.model(fef_adj, node_mask, class_label)
                loss, _, _ = self.model.module.compute_loss(
                    edge_logits, fef_adj, node_mask, mu, logvar
                )
                total_loss.append(loss.item())
            
            progress_bar.update(1)
        
        progress_bar.close()
        self.model.train()
        
        wandb.log({"Val": sum(total_loss) / len(total_loss)}, step=self.iters)
    
    def save_model(self):
        import os
        torch.save(
            self.model.module.state_dict(),
            os.path.join(self.save_dir, f'epoch_{self.epoch}.pt')
        )


# Integration function for inference
def get_topology_with_gnn(batch, gnn_model, edgeVert_model, device, labels, point_data):
    """
    Generate B-rep topology using GNN model for face-edge adjacency and 
    existing EdgeVertModel for edge-vertex adjacency.
    
    This function replaces the original get_topology function from inference/generate.py
    """
    from topology.topoGenerate import SeqGenerator
    from utils import calculate_y
    
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    vertFace_adj = []
    fef_adj = []
    
    valid = 0
    fail_idx = []
    
    if labels is not None:
        class_label = torch.LongTensor(labels).to(device).reshape(-1, 1)
    else:
        class_label = None
    
    with torch.no_grad():
        # Generate face-edge adjacency matrix using GNN
        adj_batch = gnn_model.sample(
            num_samples=batch,
            class_label=class_label,
            point_data=point_data,
            temperature=0.8
        )  # [batch, nf, nf]
        
        for i in range(batch):
            adj = adj_batch[i]  # [nf, nf]
            non_zero_mask = torch.any(adj != 0, dim=1)
            adj = adj[non_zero_mask][:, non_zero_mask]  # [nf', nf']
            
            edge_counts = torch.sum(adj, dim=1)  # nf'
            if edge_counts.max() > edgeVert_model.max_edge:
                fail_idx.append(i)
                continue
            
            sorted_ids = torch.argsort(edge_counts)
            adj = adj[sorted_ids][:, sorted_ids]
            
            edge_indices = torch.triu(adj, diagonal=1).nonzero(as_tuple=False)
            num_edges = adj[edge_indices[:, 0], edge_indices[:, 1]]
            ef_adj = edge_indices.repeat_interleave(num_edges, dim=0)  # [ne, 2]
            share_id = calculate_y(ef_adj)
            
            point_data_item = None
            if point_data is not None:
                point_data_item = point_data[i].unsqueeze(0)
            
            # Use EdgeVertModel to generate edge-vertex adjacency
            edgeVert_model.save_cache(
                edgeFace_adj=ef_adj.unsqueeze(0),
                edge_mask=torch.ones((1, ef_adj.shape[0]), device=device, dtype=torch.bool),
                share_id=share_id,
                class_label=class_label[[i]] if class_label is not None else None,
                point_data=point_data_item
            )
            
            for try_time in range(10):
                generator = SeqGenerator(ef_adj.cpu().numpy())
                if generator.generate(edgeVert_model, 
                                     class_label[[i]] if class_label is not None else None,
                                     point_data_item):
                    edgeVert_model.clear_cache()
                    valid += 1
                    break
            else:
                edgeVert_model.clear_cache()
                fail_idx.append(i)
                continue
            
            ev_adj = generator.edgeVert_adj
            fe_adj = generator.faceEdge_adj
            
            edgeVert_adj.append(torch.from_numpy(ev_adj).to(device))
            edgeFace_adj.append(ef_adj)
            faceEdge_adj.append(fe_adj)
            
            # Import helper functions
            from topology.transfer import faceVert_from_edgeVert, face_vert_trans, fef_from_faceEdge
            
            fv_adj = faceVert_from_edgeVert(fe_adj, ev_adj)
            vf_adj = face_vert_trans(faceVert_adj=fv_adj)
            vertFace_adj.append(vf_adj)
            fef = fef_from_faceEdge(edgeFace_adj=ef_adj.cpu().numpy())
            fef_adj.append(torch.from_numpy(fef).to(device))
    
    if labels is not None:
        labels = [labels[i] for i in range(len(class_label)) if i not in fail_idx]
    
    if point_data is not None:
        point_data = [point_data[i] for i in range(len(point_data)) if i not in fail_idx]
    
    print(f"Topology generation success: {valid}")
    print(f"Topology generation failed: {len(fail_idx)}")
    
    return {
        "edgeVert_adj": edgeVert_adj,
        "edgeFace_adj": edgeFace_adj,
        "faceEdge_adj": faceEdge_adj,
        "vertFace_adj": vertFace_adj,
        "fef_adj": fef_adj,
        "class_label": labels,
        "point_data": point_data,
        "fail_idx": fail_idx
    }
