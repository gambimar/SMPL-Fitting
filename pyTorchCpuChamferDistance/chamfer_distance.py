import torch

def ChamferDistanceFunction(xyz1, xyz2, use_sqrt: bool = False):
    """
    Compute the Chamfer distance between two batches of point clouds.
    
    Args:
        xyz1 (torch.Tensor): Tensor of shape (B, N, D)
        xyz2 (torch.Tensor): Tensor of shape (B, M, D)
        use_sqrt (bool): If True, returns the Euclidean distances; if False, returns squared distances.
    
    Returns:
        dist1 (torch.Tensor): For each point in xyz1, minimal distance to a point in xyz2, shape (B, N)
        dist2 (torch.Tensor): For each point in xyz2, minimal distance to a point in xyz1, shape (B, M)
        idx1 (torch.Tensor): Indices of nearest neighbors in xyz2 for each point in xyz1, shape (B, N)
        idx2 (torch.Tensor): Indices of nearest neighbors in xyz1 for each point in xyz2, shape (B, M)
    """
    # Compute pairwise Euclidean distances: shape (B, N, M)
    dists = torch.cdist(xyz1, xyz2, p=2)
    
    if not use_sqrt:
        # Use squared distances (common in many Chamfer loss implementations)
        dists = dists ** 2

    # For each point in xyz1, find the minimum distance (and index) from xyz2:
    dist1, idx1 = torch.min(dists, dim=2)  # shape: (B, N)
    # For each point in xyz2, find the minimum distance (and index) from xyz1:
    dist2, idx2 = torch.min(dists, dim=1)  # shape: (B, M)

    return dist1, dist2, idx1, idx2

class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction(xyz1, xyz2)