#!/bin/bash

# ============================================================================
# Training script for GNN-based Topology Generation
# 
# This script trains the GNN topology model that replaces the Transformer-based
# FaceEdgeModel. The GNN model uses Graph Attention Networks to generate the
# face-edge adjacency matrix.
# 
# Usage:
#   ./scripts/train_gnn.sh [dataset]
#   
#   dataset: furniture, deepcad, or abc
# ============================================================================

DATASET=${1:-deepcad}

echo "Training GNN Topology model for ${DATASET}..."

# Train GNN Topology model (replaces FaceEdge model)
python -m topology.train_gnn_topo --name ${DATASET} --batch_size 16 --train_epochs 2000

# Note: The EdgeVert model and geometry models remain the same
# You still need to train them using the original script.sh

echo "GNN Topology training complete for ${DATASET}!"
echo ""
echo "To complete the full training pipeline, also run:"
echo "  python -m topology.train_topo --name ${DATASET} --option edgeVert"
echo "  python -m geometry.train_geom --name ${DATASET} --option faceBbox"
echo "  python -m geometry.train_geom --name ${DATASET} --option vertGeom"
echo "  python -m geometry.train_geom --name ${DATASET} --option edgeGeom"
