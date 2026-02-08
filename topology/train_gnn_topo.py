"""
Training script for GNN-based Topology Generation.

This script trains the GNN topology model that replaces the Transformer-based
FaceEdgeModel for generating face-edge adjacency matrices.
"""

import os
import argparse
import wandb
import yaml
import torch
from topology.datasets import FaceEdgeDataset
from topology.gnn_topology import GNNTopologyGenerator, GNNTopologyTrainer


def get_args_gnn_topo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='deepcad',
                        choices=['furniture', 'deepcad', 'abc'])
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--train_epochs', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--test_epochs', type=int, default=50, help='number of epochs to test model')
    parser.add_argument('--save_epochs', type=int, default=200, help='number of epochs to save model')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='name of the log folder.')
    args = parser.parse_args()
    args.env = args.name + '_topo_gnn'
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    return args


def main():
    # Parse input arguments
    args = get_args_gnn_topo()
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(args.name, {})
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # Add GNN model config if not present
    if not hasattr(args, 'GNNTopologyModel'):
        args.GNNTopologyModel = {
            'd_model': 128,
            'n_layers': 4,
            'n_heads': 4,
            'dropout': 0.1
        }
    
    # Make project directory if not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Create datasets
    train_dataset = FaceEdgeDataset(os.path.join('data_process/TopoDatasets', args.name, 'train'), args)
    val_dataset = FaceEdgeDataset(os.path.join('data_process/TopoDatasets', args.name, 'test'), args)
    
    # Create trainer
    trainer = GNNTopologyTrainer(args, train_dataset, val_dataset)
    
    # Main training loop
    print('Start GNN Topology training...')
    
    # Initialize wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project='BrepGDM-GNN', dir=args.save_dir, name=args.env)
    
    # Main training loop
    for _ in range(args.train_epochs):
        # Train for one epoch
        trainer.train_one_epoch()
        
        # Evaluate model performance on validation set
        if trainer.epoch % args.test_epochs == 0:
            trainer.test_val()
        
        # Save model
        if trainer.epoch % args.save_epochs == 0:
            trainer.save_model()


if __name__ == '__main__':
    main()
