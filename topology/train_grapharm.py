"""
Training script for GraphARM topology generation.

This script trains the GraphARM model for B-rep topology generation,
replacing the sequential VAE-based approach with graph-based autoregressive diffusion.
"""

import os
import argparse
import wandb
import yaml
from topology.grapharm import GraphARMDataset
from topology.grapharm_trainer import GraphARMTrainer


def get_args():
    parser = argparse.ArgumentParser(description='Train GraphARM for B-rep topology generation')
    parser.add_argument('--name', type=str, default='furniture',
                        choices=['furniture', 'deepcad', 'abc'])
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--train_epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--test_epochs', type=int, default=50, help='number of epochs between testing')
    parser.add_argument('--save_epochs', type=int, default=200, help='number of epochs between saves')
    parser.add_argument('--dir_name', type=str, default="checkpoints", help='checkpoint directory')
    args = parser.parse_args()
    
    args.env = args.name + '_topo_grapharm'
    args.save_dir = os.path.join(args.dir_name, args.env.split('_', 1)[0], args.env.split('_', 1)[1])
    
    return args


def main():
    # Parse arguments
    args = get_args()
    
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(args.name, {})
    
    # Merge config into args
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # Add GraphARM model config
    if not hasattr(args, 'GraphARMModel'):
        args.GraphARMModel = {
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        }
    
    # Create directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Create datasets
    train_path = os.path.join('data_process/TopoDatasets', args.name, 'train')
    test_path = os.path.join('data_process/TopoDatasets', args.name, 'test')
    
    train_dataset = GraphARMDataset(train_path, args)
    val_dataset = GraphARMDataset(test_path, args)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = GraphARMTrainer(args, train_dataset, val_dataset)
    
    # Initialize wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project='BrepGDM-GraphARM', dir=args.save_dir, name=args.env)
    
    print('Start GraphARM topology training...')
    
    # Training loop
    for epoch in range(args.train_epochs):
        # Train
        train_losses = trainer.train_one_epoch()
        
        # Validate
        if trainer.epoch % args.test_epochs == 0:
            val_losses = trainer.test_val()
        
        # Save
        if trainer.epoch % args.save_epochs == 0:
            trainer.save_model()
    
    print('Training complete!')


if __name__ == '__main__':
    main()
