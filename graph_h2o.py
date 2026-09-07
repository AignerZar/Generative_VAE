import torch 
import torch.nn.functional as F

###################################################### Function for the edge index ######################################
def build_edge_index(
    P: int, 
    num_atoms: int,
    )-> torch.Tensor:
    """Function for buiding the edge indices

    Args:
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule

    Returns:
        torch.Tensor: Tensor containing the edge indices
    """
    edges = []

    for b in range(P):
        H1 = num_atoms * b + 0
        O = num_atoms * b + 1
        H2 = num_atoms * b + 2
        for i, j in [(O, H1), (H1, O), (O, H2), (H2, O)]:
            edges.append((i, j))
    
    if P > 1:
        for b in range(P):
            nb = (b + 1) % P 
            for a in range(num_atoms):
                i = num_atoms * b + a
                j = num_atoms * nb + a
                for u, v in [(i, j), (j, i)]:
                    edges.append((u, v))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

################################################ Function for the node features ####################################################
def build_node_features(
    batch_size: int, 
    P: int, 
    num_atoms: int, 
    device
    )-> torch.Tensor:
    """Function to build the node features

    Args:
        batch_size (int): Number of samples in a batch
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        device (_type_): Device the code is running on cpu or gpu

    Returns:
        torch.Tensor: Features, needed for the EGCLs
    """
    atom_types = torch.tensor([1, 0, 1], dtype=torch.long, device=device).repeat(P) 

    atom_onehot = F.one_hot(atom_types, num_classes=2).float()

    site_indices = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32, device=device).repeat(P).unsqueeze(-1)

    bead_indices = []
    for b in range(P):
        bead_indices.extend([b] * num_atoms)
    bead_indices = torch.tensor(bead_indices, dtype=torch.float32, device=device)
    
    if P > 1:
        bead_norm = (bead_indices / (P - 1)).unsqueeze(-1)
    else:
        bead_norm = torch.zeros(P * num_atoms, 1, device=device)

    base_feat = torch.cat([atom_onehot, site_indices, bead_norm], dim=-1)
    base_feat = base_feat.unsqueeze(0).expand(batch_size, P * num_atoms, base_feat.size(-1))
    return base_feat