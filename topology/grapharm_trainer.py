"""
GraphARM Trainer for B-rep Topology Generation

This module provides training functionality for the GraphARM topology model.
"""

import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from topology.grapharm import GraphARMTopologyModel, GraphARMDataset


class GraphARMTrainer:
    """
    Trainer class for GraphARM topology generation model.
    """
    def __init__(self, args, train_dataset, val_dataset):
        self.iters = 0
        self.epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = args.save_dir
        self.use_cf = args.use_cf
        self.use_pc = args.use_pc
        
        # Initialize model
        model = GraphARMTopologyModel(
            max_faces=args.max_face,
            max_edges=args.max_edge,
            edge_classes=args.edge_classes,
            hidden_dim=args.GraphARMModel.get('hidden_dim', 256),
            num_layers=args.GraphARMModel.get('num_layers', 6),
            num_heads=args.GraphARMModel.get('num_heads', 8),
            dropout=args.GraphARMModel.get('dropout', 0.1),
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
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.train_epochs, eta_min=1e-6
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_one_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")
        
        epoch_losses = {
            'total_loss': 0.0,
            'node_loss': 0.0,
            'edge_loss': 0.0,
            'ordering_loss': 0.0
        }
        num_batches = 0
        
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                data = [x.to(self.device) for x in data]
                
                if self.use_cf and self.use_pc:
                    fef_adj, mask, class_label, point_data = data
                elif self.use_cf:
                    fef_adj, mask, class_label = data
                    point_data = None
                elif self.use_pc:
                    fef_adj, mask, point_data = data
                    class_label = None
                else:
                    fef_adj, mask = data
                    class_label = None
                    point_data = None
                
                # Zero gradient
                self.optimizer.zero_grad()
                
                # Forward pass
                losses = self.model(fef_adj, mask, class_label, point_data)
                
                # Handle DataParallel output
                if isinstance(losses['total_loss'], torch.Tensor):
                    if losses['total_loss'].dim() > 0:
                        losses = {k: v.mean() for k, v in losses.items()}
                
                loss = losses['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            # Logging
            if self.iters % 20 == 0:
                wandb.log({
                    "Loss/total": losses['total_loss'].item(),
                    "Loss/node": losses['node_loss'].item(),
                    "Loss/edge": losses['edge_loss'].item(),
                    "Loss/ordering": losses['ordering_loss'].item(),
                }, step=self.iters)
                print(f"  Total: {losses['total_loss'].item():.4f}, "
                      f"Node: {losses['node_loss'].item():.4f}, "
                      f"Edge: {losses['edge_loss'].item():.4f}")
            
            self.iters += 1
            progress_bar.update(1)
        
        progress_bar.close()
        self.scheduler.step()
        self.epoch += 1
        
        # Return average losses
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def test_val(self):
        """Evaluate on validation set."""
        self.model.eval()
        
        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description("Validation")
        
        total_losses = {
            'total_loss': 0.0,
            'node_loss': 0.0,
            'edge_loss': 0.0,
            'ordering_loss': 0.0
        }
        num_batches = 0
        
        for data in self.val_dataloader:
            with torch.no_grad():
                data = [x.to(self.device) for x in data]
                
                if self.use_cf and self.use_pc:
                    fef_adj, mask, class_label, point_data = data
                elif self.use_cf:
                    fef_adj, mask, class_label = data
                    point_data = None
                elif self.use_pc:
                    fef_adj, mask, point_data = data
                    class_label = None
                else:
                    fef_adj, mask = data
                    class_label = None
                    point_data = None
                
                losses = self.model(fef_adj, mask, class_label, point_data)
                
                # Handle DataParallel output
                if isinstance(losses['total_loss'], torch.Tensor):
                    if losses['total_loss'].dim() > 0:
                        losses = {k: v.mean() for k, v in losses.items()}
                
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                num_batches += 1
            
            progress_bar.update(1)
        
        progress_bar.close()
        self.model.train()
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # Log
        wandb.log({
            "Val/total": avg_losses['total_loss'],
            "Val/node": avg_losses['node_loss'],
            "Val/edge": avg_losses['edge_loss'],
            "Val/ordering": avg_losses['ordering_loss'],
        }, step=self.iters)
        
        print(f"Validation - Total: {avg_losses['total_loss']:.4f}")
        
        return avg_losses
    
    def save_model(self):
        """Save model checkpoint."""
        save_path = os.path.join(self.save_dir, f'epoch_{self.epoch}.pt')
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def create_grapharm_datasets(args):
    """Create training and validation datasets for GraphARM."""
    from utils import load_data_with_prefix
    
    train_path = os.path.join('data_process/TopoDatasets', args.name, 'train')
    test_path = os.path.join('data_process/TopoDatasets', args.name, 'test')
    
    train_dataset = GraphARMDataset(train_path, args)
    val_dataset = GraphARMDataset(test_path, args)
    
    return train_dataset, val_dataset
