"""
Combined Inference Script for GraphARM Topology + BrepGen Geometry

This script implements the two-stage B-rep generation:
1. Stage 1: GraphARM generates topology (face-edge-face adjacency, edge-vertex adjacency)
2. Stage 2: BrepGen-style latent diffusion generates geometry

The key innovations:
- GraphARM provides structure-aware topology generation through graph diffusion
- Geometry generation uses the BrepGen approach with node duplication for tree structure
"""

import os
import numpy as np
import yaml
import torch
import torch.multiprocessing as mp
from argparse import Namespace
from tqdm import tqdm
from diffusers import DDPMScheduler, PNDMScheduler

# Import existing geometry models
from model import (
    FaceGeomTransformer, EdgeGeomTransformer, 
    VertGeomTransformer, FaceBboxTransformer
)

# Import GraphARM topology
from topology.grapharm import GraphARMTopologyModel
from topology.grapharm_generate import GraphARMTopologyGenerator, get_grapharm_topology

# Import utilities
from topology.transfer import faceVert_from_edgeVert, face_vert_trans, fef_from_faceEdge
from utils import (
    xe_mask, pad_zero, sort_bbox_multi, generate_random_string, 
    make_mask, pad_and_stack, calculate_y
)
from OCC.Extend.DataExchange import write_step_file

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Class label mappings
text2int = {
    'uncond': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4,
    'cabinet': 5, 'chair': 6, 'couch': 7, 'lamp': 8, 'sofa': 9, 'table': 10
}
int2text = {v: k for k, v in text2int.items()}


def load_grapharm_model(args, model_path):
    """Load trained GraphARM topology model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GraphARMTopologyModel(
        max_faces=args.max_face,
        max_edges=args.max_edge,
        edge_classes=args.edge_classes,
        hidden_dim=args.GraphARMModel.get('hidden_dim', 256),
        num_layers=args.GraphARMModel.get('num_layers', 6),
        num_heads=args.GraphARMModel.get('num_heads', 8),
        dropout=0.0,
        use_cf=args.use_cf,
        use_pc=args.use_pc
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    
    return model


def get_topology_grapharm(batch, grapharm_model, device, labels, point_data, args):
    """
    Generate B-rep topology using GraphARM.
    
    This replaces the sequential VAE approach with graph-based autoregressive diffusion.
    
    Args:
        batch: Number of samples to generate
        grapharm_model: Trained GraphARMTopologyModel
        device: Torch device
        labels: Optional class labels
        point_data: Optional point cloud data
        args: Configuration arguments
    
    Returns:
        Dictionary containing topology data
    """
    return get_grapharm_topology(
        batch=batch,
        model=grapharm_model,
        device=device,
        labels=labels,
        point_data=point_data,
        args=args
    )


def get_faceBbox(fef_adj, model, pndm_scheduler, ddpm_scheduler, class_label, point_data):
    """
    Generate face bounding boxes using diffusion.
    
    This is the same as in the original DTGBrepGen.
    
    Note: Import is deferred to avoid loading heavy dependencies (OCC, etc.)
    when only using the GraphARM topology module.
    """
    from inference.generate import get_faceBbox as original_get_faceBbox
    return original_get_faceBbox(fef_adj, model, pndm_scheduler, ddpm_scheduler, class_label, point_data)


def get_vertGeom(vertFace_adj, face_bbox, vertVert_adj, model, pndm_scheduler, 
                 ddpm_scheduler, class_label, point_data):
    """
    Generate vertex geometry using diffusion.
    
    This is the same as in the original DTGBrepGen.
    
    Note: Import is deferred to avoid loading heavy dependencies (OCC, etc.)
    when only using the GraphARM topology module.
    """
    from inference.generate import get_vertGeom as original_get_vertGeom
    return original_get_vertGeom(vertFace_adj, face_bbox, vertVert_adj, model, 
                                  pndm_scheduler, ddpm_scheduler, class_label, point_data)


def get_edgeGeom(edgeFace_bbox, edgeVert_geom, edge_mask, model, pndm_scheduler,
                 ddpm_scheduler, class_label, point_data):
    """
    Generate edge geometry using diffusion.
    
    This is the same as in the original DTGBrepGen.
    
    Note: Import is deferred to avoid loading heavy dependencies (OCC, etc.)
    when only using the GraphARM topology module.
    """
    from inference.generate import get_edgeGeom as original_get_edgeGeom
    return original_get_edgeGeom(edgeFace_bbox, edgeVert_geom, edge_mask, model,
                                  pndm_scheduler, ddpm_scheduler, class_label, point_data)


def get_faceGeom(face_bbox, faceVert_geom, faceEdge_geom, face_mask, faceVert_mask,
                 faceEdge_mask, model, pndm_scheduler, ddpm_scheduler, class_label, point_data):
    """
    Generate face geometry using diffusion.
    
    This is the same as in the original DTGBrepGen.
    
    Note: Import is deferred to avoid loading heavy dependencies (OCC, etc.)
    when only using the GraphARM topology module.
    """
    from inference.generate import get_faceGeom as original_get_faceGeom
    return original_get_faceGeom(face_bbox, faceVert_geom, faceEdge_geom, face_mask,
                                  faceVert_mask, faceEdge_mask, model, pndm_scheduler,
                                  ddpm_scheduler, class_label, point_data)


def generate_brep_grapharm(args, num_samples=10, class_labels=None, point_data=None,
                            save_dir='samples/grapharm'):
    """
    Main function to generate B-rep models using GraphARM topology + BrepGen geometry.
    
    Args:
        args: Configuration arguments
        num_samples: Number of samples to generate
        class_labels: Optional list of class labels
        point_data: Optional point cloud data [num_samples, 3, 2000]
        save_dir: Directory to save generated models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("GraphARM Topology + BrepGen Geometry Generation")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # ==================== Load Models ====================
    print("\nLoading models...")
    
    # GraphARM topology model
    grapharm_path = os.path.join('checkpoints', args.name, 'topo_grapharm/epoch_2000.pt')
    if not os.path.exists(grapharm_path):
        grapharm_path = os.path.join('checkpoints', args.name, 'topo_grapharm/epoch_1000.pt')
    grapharm_model = load_grapharm_model(args, grapharm_path)
    print(f"  - GraphARM topology model loaded from {grapharm_path}")
    
    # Geometry models
    faceBbox_model = FaceBboxTransformer(
        n_layers=args.FaceBboxModel['n_layers'],
        hidden_mlp_dims=args.FaceBboxModel['hidden_mlp_dims'],
        hidden_dims=args.FaceBboxModel['hidden_dims'],
        edge_classes=args.edge_classes,
        act_fn_in=torch.nn.ReLU(),
        act_fn_out=torch.nn.ReLU(),
        use_cf=args.use_cf,
        use_pc=args.use_pc
    )
    faceBbox_path = os.path.join('checkpoints', args.name, 'geom_faceBbox/epoch_3000.pt')
    faceBbox_model.load_state_dict(torch.load(faceBbox_path, map_location=device))
    faceBbox_model = faceBbox_model.to(device).eval()
    print(f"  - FaceBbox model loaded from {faceBbox_path}")
    
    vertGeom_model = VertGeomTransformer(
        n_layers=args.VertGeomModel['n_layers'],
        hidden_mlp_dims=args.VertGeomModel['hidden_mlp_dims'],
        hidden_dims=args.VertGeomModel['hidden_dims'],
        act_fn_in=torch.nn.ReLU(),
        act_fn_out=torch.nn.ReLU(),
        use_cf=args.use_cf,
        use_pc=args.use_pc
    )
    vertGeom_path = os.path.join('checkpoints', args.name, 'geom_vertGeom/epoch_3000.pt')
    vertGeom_model.load_state_dict(torch.load(vertGeom_path, map_location=device))
    vertGeom_model = vertGeom_model.to(device).eval()
    print(f"  - VertGeom model loaded from {vertGeom_path}")
    
    edgeGeom_model = EdgeGeomTransformer(
        n_layers=args.EdgeGeomModel['n_layers'],
        edge_geom_dim=args.EdgeGeomModel['edge_geom_dim'],
        d_model=args.EdgeGeomModel['d_model'],
        nhead=args.EdgeGeomModel['nhead'],
        use_cf=args.use_cf,
        use_pc=args.use_pc
    )
    edgeGeom_path = os.path.join('checkpoints', args.name, 'geom_edgeGeom/epoch_3000.pt')
    edgeGeom_model.load_state_dict(torch.load(edgeGeom_path, map_location=device))
    edgeGeom_model = edgeGeom_model.to(device).eval()
    print(f"  - EdgeGeom model loaded from {edgeGeom_path}")
    
    faceGeom_model = FaceGeomTransformer(
        n_layers=args.FaceGeomModel['n_layers'],
        face_geom_dim=args.FaceGeomModel['face_geom_dim'],
        d_model=args.FaceGeomModel['d_model'],
        nhead=args.FaceGeomModel['nhead'],
        use_cf=args.use_cf,
        use_pc=args.use_pc
    )
    faceGeom_path = os.path.join('checkpoints', args.name, 'geom_faceGeom/epoch_3000.pt')
    faceGeom_model.load_state_dict(torch.load(faceGeom_path, map_location=device))
    faceGeom_model = faceGeom_model.to(device).eval()
    print(f"  - FaceGeom model loaded from {faceGeom_path}")
    
    # Initialize schedulers
    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        skip_prk_steps=True,
        set_alpha_to_one=False
    )
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        clip_sample=True
    )
    
    print("\n" + "=" * 60)
    print("Stage 1: GraphARM Topology Generation")
    print("=" * 60)
    
    # ==================== Generate Topology ====================
    topo_data = get_topology_grapharm(
        batch=num_samples,
        grapharm_model=grapharm_model,
        device=device,
        labels=class_labels,
        point_data=point_data,
        args=args
    )
    
    valid_samples = num_samples - len(topo_data['fail_idx'])
    print(f"  Successfully generated {valid_samples}/{num_samples} topologies")
    
    if valid_samples == 0:
        print("No valid topologies generated. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("Stage 2: BrepGen-style Geometry Generation")
    print("=" * 60)
    
    # ==================== Generate Face Bounding Boxes ====================
    print("\n[2.1] Generating face bounding boxes...")
    face_bbox, face_mask = get_faceBbox(
        fef_adj=topo_data['fef_adj'],
        model=faceBbox_model,
        pndm_scheduler=pndm_scheduler,
        ddpm_scheduler=ddpm_scheduler,
        class_label=topo_data['class_label'],
        point_data=topo_data['point_data']
    )
    print(f"  Face bbox shape: {face_bbox.shape}")
    
    # ==================== Generate Vertex Geometry ====================
    print("\n[2.2] Generating vertex geometry...")
    from topology.transfer import compute_vertVert_adj
    
    vertVert_adj = []
    for ev_adj in topo_data['edgeVert_adj']:
        vv_adj = compute_vertVert_adj(ev_adj.cpu().numpy())
        vertVert_adj.append(torch.from_numpy(vv_adj).to(device))
    
    vert_geom, vert_mask = get_vertGeom(
        vertFace_adj=topo_data['vertFace_adj'],
        face_bbox=face_bbox,
        vertVert_adj=vertVert_adj,
        model=vertGeom_model,
        pndm_scheduler=pndm_scheduler,
        ddpm_scheduler=ddpm_scheduler,
        class_label=topo_data['class_label'],
        point_data=topo_data['point_data']
    )
    print(f"  Vertex geom shape: {vert_geom.shape}")
    
    # ==================== Generate Edge Geometry ====================
    print("\n[2.3] Generating edge geometry...")
    
    # Prepare edge data
    edgeFace_bbox = []
    edgeVert_geom = []
    edge_masks = []
    
    for i in range(valid_samples):
        ef_adj = topo_data['edgeFace_adj'][i]
        ev_adj = topo_data['edgeVert_adj'][i]
        
        # Get face bboxes for each edge's adjacent faces
        ef_bbox = face_bbox[i][ef_adj]  # [ne, 2, 6]
        edgeFace_bbox.append(ef_bbox)
        
        # Get vertex geom for each edge's vertices
        ev_geom = vert_geom[i][ev_adj]  # [ne, 2, 3]
        edgeVert_geom.append(ev_geom)
        
        edge_masks.append(torch.ones(len(ef_adj), dtype=torch.bool, device=device))
    
    # Pad and stack
    edgeFace_bbox, _ = pad_and_stack(edgeFace_bbox)
    edgeVert_geom, edge_mask = pad_and_stack(edgeVert_geom)
    
    edge_geom = get_edgeGeom(
        edgeFace_bbox=edgeFace_bbox,
        edgeVert_geom=edgeVert_geom,
        edge_mask=edge_mask,
        model=edgeGeom_model,
        pndm_scheduler=pndm_scheduler,
        ddpm_scheduler=ddpm_scheduler,
        class_label=topo_data['class_label'],
        point_data=topo_data['point_data']
    )
    print(f"  Edge geom shape: {edge_geom.shape}")
    
    # ==================== Generate Face Geometry ====================
    print("\n[2.4] Generating face geometry...")
    
    # Prepare face data
    faceVert_geom = []
    faceEdge_geom = []
    faceVert_masks = []
    faceEdge_masks = []
    
    for i in range(valid_samples):
        fe_adj = topo_data['faceEdge_adj'][i]
        ev_adj = topo_data['edgeVert_adj'][i]
        
        # Get vertex geom for each face
        fv_geom = []
        fv_mask = []
        fe_geom = []
        fe_mask = []
        
        for face_edges in fe_adj:
            # Get vertices of this face
            face_verts = set()
            for e in face_edges:
                face_verts.update(ev_adj[e].tolist())
            face_verts = list(face_verts)
            
            fv_geom.append(vert_geom[i][face_verts])  # [fv, 3]
            fv_mask.append(len(face_verts))
            
            fe_geom.append(edge_geom[i][face_edges])  # [fe, 12]
            fe_mask.append(len(face_edges))
        
        faceVert_geom.append(fv_geom)
        faceEdge_geom.append(fe_geom)
        faceVert_masks.append(fv_mask)
        faceEdge_masks.append(fe_mask)
    
    face_geom = get_faceGeom(
        face_bbox=face_bbox[:valid_samples],
        faceVert_geom=faceVert_geom,
        faceEdge_geom=faceEdge_geom,
        face_mask=face_mask[:valid_samples],
        faceVert_mask=faceVert_masks,
        faceEdge_mask=faceEdge_masks,
        model=faceGeom_model,
        pndm_scheduler=pndm_scheduler,
        ddpm_scheduler=ddpm_scheduler,
        class_label=topo_data['class_label'],
        point_data=topo_data['point_data']
    )
    print(f"  Face geom shape: {face_geom.shape}")
    
    print("\n" + "=" * 60)
    print("Stage 3: B-rep Construction")
    print("=" * 60)
    
    # ==================== Construct B-rep Models ====================
    from inference.brepBuild import construct_brep
    
    success_count = 0
    for i in tqdm(range(valid_samples), desc="Constructing B-rep"):
        try:
            # Prepare data for B-rep construction
            data = {
                'edgeVert_adj': topo_data['edgeVert_adj'][i].cpu().numpy(),
                'edgeFace_adj': topo_data['edgeFace_adj'][i].cpu().numpy(),
                'faceEdge_adj': topo_data['faceEdge_adj'][i],
                'vert_geom': vert_geom[i].cpu().numpy(),
                'edge_geom': edge_geom[i].cpu().numpy(),
                'face_geom': face_geom[i].cpu().numpy(),
                'face_bbox': face_bbox[i].cpu().numpy(),
            }
            
            # Construct B-rep
            brep = construct_brep(data)
            
            if brep is not None:
                # Save to STEP file
                filename = f"sample_{i:04d}"
                if topo_data['class_label'] is not None:
                    label = topo_data['class_label'][i]
                    if isinstance(label, int) and label in int2text:
                        filename = f"{int2text[label]}_{i:04d}"
                
                filepath = os.path.join(save_dir, f"{filename}.step")
                write_step_file(brep, filepath)
                success_count += 1
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print(f"Generation Complete!")
    print(f"  Successfully generated: {success_count}/{valid_samples} B-rep models")
    print(f"  Saved to: {save_dir}")
    print(f"{'=' * 60}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate B-rep using GraphARM + BrepGen')
    parser.add_argument('--name', type=str, default='furniture',
                        choices=['furniture', 'deepcad', 'abc'])
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--save_dir', type=str, default='samples/grapharm', help='Output directory')
    parser.add_argument('--class_label', type=str, default=None, help='Class label (for conditional generation)')
    args = parser.parse_args()
    
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(args.name, {})
    
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # Parse class label
    class_labels = None
    if args.class_label is not None:
        if args.class_label in text2int:
            class_labels = [text2int[args.class_label]] * args.num_samples
        else:
            try:
                class_labels = [int(args.class_label)] * args.num_samples
            except ValueError:
                print(f"Invalid class label: {args.class_label}")
                return
    
    # Generate
    generate_brep_grapharm(
        args=args,
        num_samples=args.num_samples,
        class_labels=class_labels,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
