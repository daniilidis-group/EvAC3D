import torch
import numpy as np

def random_transform_like(X, scale_min=0.2, scale_max=0.5, max_dist=1.0):
    return random_transform(X.shape[0], scale_min, scale_max, max_dist, X.device)

def random_transform(batch_size, scale_min=0.25, scale_max=0.65, max_dist=1.0, device=None):
    """
    - batch_size - number of random transformations to make
    - scale_min - minimum scale for S
    - scale_max - maximum scale for S
    - max_dist - the largest distance in x or y to be away from the origin
    - device - device to create this transform on
    """

    scale = torch.rand(batch_size, 1, 1, device=device)
    scale = (scale_max-scale_min) * scale + scale_min
    S = scale*torch.eye(2, device=device)[None,...].expand(-1,2,2)

    trans = 2 * max_dist * (torch.rand(batch_size, 2, 1, device=device)-0.5)

    phi = 2. * np.pi * (torch.rand(batch_size, 1, device=device)-0.5)

    R = torch.stack([torch.cos(phi), -torch.sin(phi), torch.sin(phi), torch.cos(phi)], dim=-1)
    R = R.reshape(-1,2,2)

    return torch.cat([torch.bmm(S, R), trans], dim=-1).detach()

def apply_transform(A_delta, A):
    # Is there a more efficient way of implementing this?
    A_out = torch.cat([ torch.matmul(A_delta[:,:2,:2], A[:,:2,:2]),
                        torch.matmul(A_delta[:,:2,:2], A[:,:2,2,None]) + A_delta[:,:2,2,None] ], 2)
    return A_out

