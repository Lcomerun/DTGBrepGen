"""
GraphARM Topology Generation Module

This module provides functionality for generating B-rep topology using the
trained GraphARM model. It converts the generated face adjacency into
the full topology structure (edgeFace_adj, edgeVert_adj, faceEdge_adj, etc.)
"""

import os
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from topology.grapharm import GraphARMTopologyModel
from topology.datasets import compute_topoSeq
from topology.transfer import faceVert_from_edgeVert, face_vert_trans, fef_from_faceEdge


class GraphARMTopologyGenerator:
    """
    Generator for B-rep topology using GraphARM.
    
    This class handles:
    1. Loading a trained GraphARM model
    2. Generating face-edge-face adjacency matrices
    3. Converting to full B-rep topology (edgeFace_adj, edgeVert_adj, etc.)
    """
    
    def __init__(self, args, model_path: str):
        """
        Initialize the topology generator.
        
        Args:
            args: Configuration arguments
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        
        # Initialize model
        self.model = GraphARMTopologyModel(
            max_faces=args.max_face,
            max_edges=args.max_edge,
            edge_classes=args.edge_classes,
            hidden_dim=args.GraphARMModel.get('hidden_dim', 256),
            num_layers=args.GraphARMModel.get('num_layers', 6),
            num_heads=args.GraphARMModel.get('num_heads', 8),
            dropout=0.0,  # No dropout during inference
            use_cf=args.use_cf,
            use_pc=args.use_pc
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()
    
    def generate_fef_adj(self, num_samples: int, num_faces: Optional[int] = None,
                         class_label: Optional[torch.Tensor] = None,
                         point_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate face-edge-face adjacency matrices.
        
        Args:
            num_samples: Number of samples to generate
            num_faces: Number of faces (if None, randomly chosen)
            class_label: Optional class conditions [num_samples, 1]
            point_data: Optional point cloud conditions [num_samples, 3, 2000]
        
        Returns:
            fef_adj: Face adjacency matrices [num_samples, max_faces, max_faces]
        """
        with torch.no_grad():
            fef_adj = self.model.sample(
                num_samples=num_samples,
                num_faces=num_faces,
                class_label=class_label,
                point_data=point_data
            )
        return fef_adj
    
    def fef_to_edgeFace(self, fef_adj: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Convert face-edge-face adjacency to edge-face adjacency.
        
        The fef_adj encodes how many edges connect each pair of faces.
        We expand this to create individual edges.
        
        Args:
            fef_adj: Face adjacency [num_faces, num_faces]
        
        Returns:
            edgeFace_adj: Edge to face mapping [num_edges, 2]
            num_faces: Number of valid faces
        """
        device = fef_adj.device
        
        # Remove empty rows/columns (invalid faces)
        valid_mask = torch.any(fef_adj != 0, dim=1)
        fef_adj = fef_adj[valid_mask][:, valid_mask]
        num_faces = fef_adj.shape[0]
        
        # Find non-zero upper triangle entries
        edge_indices = torch.triu(fef_adj, diagonal=1).nonzero(as_tuple=False)
        num_edges_per_pair = fef_adj[edge_indices[:, 0], edge_indices[:, 1]]
        
        # Expand to individual edges
        edgeFace_adj = edge_indices.repeat_interleave(num_edges_per_pair.long(), dim=0)
        
        return edgeFace_adj, num_faces
    
    def generate_edge_vertex_topology(self, edgeFace_adj: np.ndarray, 
                                       faceEdge_adj: List[List[int]]) -> Optional[np.ndarray]:
        """
        Generate edge-vertex adjacency that satisfies topological constraints.
        
        This implements a constraint satisfaction approach to assign vertices
        to edges such that:
        1. Each edge connects exactly 2 vertices
        2. Each face forms a closed loop
        3. Adjacent edges share vertices
        
        Args:
            edgeFace_adj: Edge to face mapping [num_edges, 2]
            faceEdge_adj: Face to edges mapping List[List[int]]
        
        Returns:
            edgeVert_adj: Edge to vertex mapping [num_edges, 2] or None if failed
        """
        num_edges = edgeFace_adj.shape[0]
        num_faces = len(faceEdge_adj)
        
        # Initialize vertex assignment
        # Each edge has 2 "corners" (2*edge_id and 2*edge_id+1)
        # We need to merge corners that share the same vertex
        vert_flag = np.arange(2 * num_edges)  # Each corner starts as its own vertex
        set_flag = {i: {i} for i in range(2 * num_edges)}
        
        def find_set(v):
            return vert_flag[v]
        
        def merge_sets(v1, v2):
            s1, s2 = find_set(v1), find_set(v2)
            if s1 == s2:
                return True
            
            # Check constraint: two corners of same edge can't merge
            merged = set_flag[s1] | set_flag[s2]
            for c in merged:
                if c % 2 == 0 and (c + 1) in merged:
                    return False
                if c % 2 == 1 and (c - 1) in merged:
                    return False
            
            # Merge smaller set into larger
            if len(set_flag[s1]) < len(set_flag[s2]):
                s1, s2 = s2, s1
            
            set_flag[s1].update(set_flag[s2])
            for c in set_flag[s2]:
                vert_flag[c] = s1
            del set_flag[s2]
            return True
        
        # Process each face to create closed loops
        for face_id, face_edges in enumerate(faceEdge_adj):
            if len(face_edges) < 2:
                continue
            
            # Connect consecutive edges in the face loop
            for i in range(len(face_edges)):
                edge1 = face_edges[i]
                edge2 = face_edges[(i + 1) % len(face_edges)]
                
                # Try to find a valid corner pairing
                corners1 = [2 * edge1, 2 * edge1 + 1]
                corners2 = [2 * edge2, 2 * edge2 + 1]
                
                merged = False
                for c1 in corners1:
                    for c2 in corners2:
                        if merge_sets(c1, c2):
                            merged = True
                            break
                    if merged:
                        break
        
        # Build edgeVert_adj from final vertex assignments
        # Renumber vertices to be consecutive
        unique_sets = list(set_flag.keys())
        vert_map = {s: i for i, s in enumerate(unique_sets)}
        
        edgeVert_adj = np.zeros((num_edges, 2), dtype=np.int64)
        for e in range(num_edges):
            c1, c2 = 2 * e, 2 * e + 1
            v1 = vert_map[find_set(c1)]
            v2 = vert_map[find_set(c2)]
            edgeVert_adj[e] = [v1, v2]
        
        # Verify topology
        if not self._verify_topology(edgeVert_adj, faceEdge_adj):
            return None
        
        return edgeVert_adj
    
    def _verify_topology(self, edgeVert_adj: np.ndarray, 
                         faceEdge_adj: List[List[int]]) -> bool:
        """Verify that the generated topology is valid."""
        # Check: two vertices on same edge must be different
        if not np.all(edgeVert_adj[:, 0] != edgeVert_adj[:, 1]):
            return False
        
        # Check: each vertex connects to at least 2 edges
        num_verts = edgeVert_adj.max() + 1
        vert_count = np.zeros(num_verts)
        np.add.at(vert_count, edgeVert_adj.flatten(), 1)
        if np.any(vert_count < 2):
            return False
        
        # Check: each face has equal edges and vertices
        for face_edges in faceEdge_adj:
            verts = set()
            for e in face_edges:
                verts.update(edgeVert_adj[e])
            if len(face_edges) != len(verts):
                return False
        
        return True
    
    def generate_full_topology(self, num_samples: int, 
                               class_labels: Optional[List[int]] = None,
                               point_data: Optional[torch.Tensor] = None,
                               max_retries: int = 10) -> Dict:
        """
        Generate complete B-rep topology structures.
        
        Args:
            num_samples: Number of samples to generate
            class_labels: Optional list of class labels
            point_data: Optional point cloud data
            max_retries: Maximum retries per sample
        
        Returns:
            Dictionary containing:
            - edgeVert_adj: List of [num_edges, 2] arrays
            - edgeFace_adj: List of [num_edges, 2] tensors
            - faceEdge_adj: List of List[List[int]]
            - vertFace_adj: List of vertex-face adjacencies
            - fef_adj: List of [num_faces, num_faces] tensors
            - class_label: List of class labels (if provided)
            - point_data: Point data (if provided)
            - fail_idx: Indices of failed samples
        """
        results = {
            'edgeVert_adj': [],
            'edgeFace_adj': [],
            'faceEdge_adj': [],
            'vertFace_adj': [],
            'fef_adj': [],
            'class_label': class_labels.copy() if class_labels else None,
            'point_data': point_data,
            'fail_idx': []
        }
        
        # Prepare class labels
        if class_labels is not None:
            class_label_tensor = torch.LongTensor(class_labels).to(self.device).reshape(-1, 1)
        else:
            class_label_tensor = None
        
        if point_data is not None:
            point_data = point_data.to(self.device)
        
        # Generate face adjacencies
        fef_adj_batch = self.generate_fef_adj(
            num_samples=num_samples,
            class_label=class_label_tensor,
            point_data=point_data
        )
        
        valid_count = 0
        
        for i in tqdm(range(num_samples), desc="Building topology"):
            fef_adj = fef_adj_batch[i]
            
            success = False
            for retry in range(max_retries):
                try:
                    # Convert to edgeFace adjacency
                    edgeFace_adj, num_faces = self.fef_to_edgeFace(fef_adj)
                    
                    if len(edgeFace_adj) == 0:
                        continue
                    
                    # Build faceEdge adjacency
                    faceEdge_adj = [[] for _ in range(num_faces)]
                    for edge_idx, (f1, f2) in enumerate(edgeFace_adj.cpu().numpy()):
                        faceEdge_adj[f1].append(edge_idx)
                        faceEdge_adj[f2].append(edge_idx)
                    
                    # Check max edges per face
                    max_edges_per_face = max(len(edges) for edges in faceEdge_adj)
                    if max_edges_per_face > self.args.max_edge:
                        continue
                    
                    # Generate edge-vertex topology
                    edgeVert_adj = self.generate_edge_vertex_topology(
                        edgeFace_adj.cpu().numpy(), 
                        faceEdge_adj
                    )
                    
                    if edgeVert_adj is None:
                        # Retry with new sample
                        if retry < max_retries - 1:
                            fef_adj = self.generate_fef_adj(
                                num_samples=1,
                                class_label=class_label_tensor[[i]] if class_label_tensor is not None else None,
                                point_data=point_data[[i]] if point_data is not None else None
                            )[0]
                        continue
                    
                    # Build other adjacencies
                    faceVert_adj = faceVert_from_edgeVert(faceEdge_adj, edgeVert_adj)
                    vertFace_adj = face_vert_trans(faceVert_adj=faceVert_adj)
                    
                    # Store results
                    results['edgeVert_adj'].append(torch.from_numpy(edgeVert_adj).to(self.device))
                    results['edgeFace_adj'].append(edgeFace_adj)
                    results['faceEdge_adj'].append(faceEdge_adj)
                    results['vertFace_adj'].append(vertFace_adj)
                    
                    # Rebuild fef for consistency
                    fef = fef_from_faceEdge(edgeFace_adj=edgeFace_adj.cpu().numpy())
                    results['fef_adj'].append(torch.from_numpy(fef).to(self.device))
                    
                    success = True
                    valid_count += 1
                    break
                    
                except Exception as e:
                    continue
            
            if not success:
                results['fail_idx'].append(i)
        
        # Filter out failed samples from labels
        if results['class_label'] is not None:
            results['class_label'] = [
                results['class_label'][i] 
                for i in range(num_samples) 
                if i not in results['fail_idx']
            ]
        
        print(f"Topology generation: {valid_count}/{num_samples} successful")
        
        return results


def get_grapharm_topology(batch: int, model: GraphARMTopologyModel, device: torch.device,
                          labels: Optional[List[int]] = None, 
                          point_data: Optional[torch.Tensor] = None,
                          args=None) -> Dict:
    """
    Convenience function to generate topology using GraphARM.
    
    This function provides the same interface as get_topology() in generate.py
    but uses GraphARM instead of the sequential VAE approach.
    
    Args:
        batch: Number of samples to generate
        model: Trained GraphARMTopologyModel
        device: Torch device
        labels: Optional class labels
        point_data: Optional point cloud data
        args: Configuration arguments
    
    Returns:
        Dictionary with topology data
    """
    generator = GraphARMTopologyGenerator.__new__(GraphARMTopologyGenerator)
    generator.device = device
    generator.model = model
    generator.args = args
    
    return generator.generate_full_topology(
        num_samples=batch,
        class_labels=labels,
        point_data=point_data
    )
