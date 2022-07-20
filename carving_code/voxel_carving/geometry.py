import torch

import numpy as np

from .utils import _device, INF, normalize, for_every, normalized_clip

import pdb

def get_identity():
    return torch.eye(4, device=_device)

def transform_rays(origin, direction, T):
    """
    Apply T to the rays defined by origin and direction
    """
    no = (origin @ T[:3,:3].T) + T[:3,3]
    nd = direction @ T[:3,:3].T
    return no, nd

def invert_transform(T):
    nT = torch.eye(4)
    nT[:3,:3] = T[:3,:3].T
    nT[:3,3] = nT[:3,:3] @ (-1.*T[:3,3])
    return nT

def apply_global_velocity(T, x, y, z, r, p, yaw, use_numpy=True):
    if use_numpy:
        nT = np.eye(4)
    else:
        nT = get_identity()

    nT[0,3] = T[0,3] + x
    nT[1,3] = T[1,3] + y
    nT[2,3] = T[2,3] + z

    nT[:3,:3] = rpy_to_tensor(r, p, yaw) @ T[:3,:3]

    return nT

def rpy_to_tensor(r, p, y):
    Rx = np.array([[1., 0., 0.],
                       [0., np.cos(r),-np.sin(r)],
                       [0., np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0., np.sin(p)],
                       [0., 1., 0.],
                       [-np.sin(p), 0., np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y), 0.],
                       [np.sin(y), np.cos(y), 0.],
                       [0.,0.,1.]])
    return Rz @ Ry @ Rx

def dist_from_ray(O,D,M):
    """
    Arguments:
    - O : (nR,3) - numRays of ray origins
    - D : (nR,3) - numRays of ray directions
    - M : (nP,3) - numPoints to check

    Returns:
    - distances : the distance from every ray to every point
    """

    O = O[:,None,:].expand(-1,M.shape[0],-1)
    D = D[:,None,:].expand(-1,M.shape[0],-1)
    M = M[None,:,:].expand(O.shape[0],-1,-1)

    norm_lines = torch.norm(D, dim=-1)
    cross_result = torch.cross(D, M-O, dim=-1)
    norm_cross = torch.norm(cross_result, dim=-1)

    return norm_cross / norm_lines

def dist_from_plane(P,N,M):
    """
    Arguments:
    - P : (nP,3) - numPlanes of plane origins
    - N : (nP,3) - numPlanes of plane normals
    - M : (nP,3) - numPoints to check
    """

    P = P[:,None,:].expand(-1,M.shape[0],-1)
    N = N[:,None,:].expand(-1,M.shape[0],-1)
    M = M[None,:,:].expand(P.shape[0],-1,-1)

    NdotP = torch.sum(N*P, dim=-1)
    NdotM = torch.sum(N*M, dim=-1)

    norm_plane = torch.norm(N, dim=-1)

    return torch.abs(NdotM - NdotP) / norm_plane

def ray_intersect_plane(O,D,P,N):
    """
    Arguments:
    - O : (nR,3) - numRays of ray origins
    - D : (nR,3) - numRays of directions
    - P : (nP,3) - numPlanes of plane origins
    - N : (nP,3) - numPlanes of plane normals

    Returns:
    - distances : The distance from every ray origin to every plane (nR,nP)

    Checks if the ray(s) defined by O,D intersect the plane(s) defined by P,N

    We first expand all to become (nR,nP,3) as we want to compute for every combination
    of rays and planes.
    """

    O = O[:,None,:] # Ray variable so add in plane dim
    D = D[:,None,:] # Ray variable so add in plane dim
    P = P[None,:,:] # Plane variable so add in ray dim
    N = N[None,:,:] # Plane variable so add in ray dim

    denom = torch.sum(D * N, dim=-1)

    distances = torch.sum( (P-O) * N, dim=-1) / denom

    distances[denom.abs() < 1e-6] = INF
    distances[distances < 0.] = INF

    return distances # (nR,nP)

def ray_intersect_sphere(O, D, S, R):
    """
    Arguments:
    - O : (nR,3) - numRays of ray origins
    - D : (nR,3) - numRays of directions
    - S : (nS,3) - numSpheres of sphere origins
    - R : (nS,3) - numSpheres

    Returns:
    - distances : The distance from every ray origin to every plane (nR,nS)

    Checks if the ray(s) defined by O,D intersect the spheres(s) defined by S, R
    """

    O = O[:,None,:]
    D = D[:,None,:]
    S = S[None,:,:]
    R = R[None,:]

    a = torch.sum(D*D,dim=-1)
    OS = O - S
    b = 2 * torch.sum(D*OS,dim=-1)
    c = torch.sum(OS*OS,dim=-1) - R*R

    disc = b * b - 4 * a * c

    distSqrt = torch.sqrt(disc)
    q = torch.where( b<0, (-b-distSqrt)/2.0, (-b+distSqrt)/2.0 )
    t0 = q / a
    t1 = c / q

    p0 = torch.where(t0<t1,t0,t1)
    p1 = torch.where(t0>t1,t0,t1)

    results = torch.where(p0<0.,p1,p0)

    results[t1<0.] = INF
    results[disc<=0.] = INF
    
    return results

def intersect(rayO, rayD, og):
    """
    Arguments:
    - rayO : (nR,3) or (3,) - the origin of the rays
    - rayD : (nR,3) - the direction of the rays
    - og : Object group as defined by the individual intersects
    """

    obj_type = og['type']

    if obj_type == 'spheres':
        return ray_intersect_sphere(rayO, rayD, og['position'], og['r'])
    elif obj_type == 'planes':
        return ray_intersect_plane(rayO, rayD, og['position'], og['normal'])
    else:
        raise NotImplementedError("Object type requested not implemented.")

def get_normal_sphere(M, positions):
    """
    Arguments:
    - M - (nR,3) - intersection points
    - normals - (nP,3) - normals of the planes in the scene

    Returns:
    - (nR,nP,3) - the normal of every plane on every sphere
    """

    M = M[:,None,:]
    positions = positions[None,:,:]
    return normalize(M - positions)

def get_normal_plane(M, normals):
    """
    Arguments:
    - M - (nR,3) - intersection points
    - normals - (nP,3) - normals of the planes in the scene

    Returns:
    - (nR,nP,3) - the normals of the planes as each ray would hit
    """
    normals = normals[None,:,:]
    return normals.expand((M.shape[0], -1, -1))

def get_normal(M, og):
    obj_type = og['type']

    if obj_type == 'spheres':
        return get_normal_sphere(M, og['position'])
    elif obj_type == 'planes':
        return get_normal_plane(M, og['normal'])
    else:
        raise NotImplementedError("Object type requested not implemented.")

def get_color(M, og):
    color = og['color'][None,:,:]
    return color.expand((M.shape[0],-1,-1))

def trace_ray_depth(rayO, rayD, env):
    """
    For all rays defined, trace each to the first object that it hits.

    Arguments:
    - rayO : (nR,3) or (3,) - the origin of the rays
    - rayD : (nR,3) - the direction of the rays
    """

    objects = env.object_groups

    rayDshape = rayD.shape

    rayD = rayD.reshape(-1,3)
    if len(rayO.shape) == 1:
        rayO = rayO.reshape(1,3).expand(rayD.shape[0],-1)
    rayO = rayO.reshape(-1,3)

    env_depths = []

    for object_group in objects:
        env_depths.append( intersect(rayO, rayD, object_group) )

    depths_per_object = torch.cat(env_depths, dim=-1)
    depths, idx = torch.min(depths_per_object, dim=-1)

    return depths.reshape(*rayDshape[:-1])

def trace_ray_color(rayO, rayD, env, max_bounces=1):
    if max_bounces > 1:
        raise NotImplementedError("More complex handling must be done for multiple bounces")

    objects = env.object_groups
    lights = env.lights

    rayDshape = rayD.shape

    rayD = rayD.reshape(-1,3)
    if len(rayO.shape) == 1:
        rayO = rayO.reshape(1,3).expand(rayD.shape[0],-1)
    rayO = rayO.reshape(-1,3)

    # Get depths for all possible objects
    env_depths = []
    for object_group in objects:
        env_depths.append( intersect(rayO, rayD, object_group) )
    depths_per_object = torch.cat(env_depths, dim=-1)

    depths_per_object = for_every(objects, intersect, rayO, rayD)
    # Reduce to first object for each pixel
    depths, idx = torch.min(depths_per_object, dim=-1, keepdim=True)

    # Get object intersection
    M = rayO + rayD * depths

    # Get normals for every object
    normal_per_object = for_every(objects, get_normal, M, cat_dim=1)
    N = torch.gather( normal_per_object, 1, idx[:,:,None].expand(-1,-1,3) ).squeeze()

    # Get colors for every object
    color_per_object = for_every(objects, get_color, M, cat_dim=1)
    C = torch.gather( color_per_object, 1, idx[:,:,None].expand(-1,-1,color_per_object.shape[-1]) ).squeeze()

    # Compute the lambertian shading for each light source
    lambert_per_light = for_every(lights, 'lambertian', C, N, M, cat_dim=1, function_of_object=True)

    # Compute the Blinn-Phong shading for each light source
    specular_per_light = for_every(lights, 'specular', C, rayO, N, M, cat_dim=1, function_of_object=True)

    color = lambert_per_light.sum(dim=1) + specular_per_light.sum(dim=1)

    color = normalized_clip(color)

    return color.reshape(*rayDshape[:-1],4)
