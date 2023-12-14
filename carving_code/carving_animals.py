#from o3d_voxel_example import open3d_tutorial as o3dtut
import open3d as o3d
import numpy as np
import os
import mcubes
import torch
import h5py

import pdb

from voxel_carving import ContinuousCamera, Volume, normalize, transform_rays, invert_transform, EnvironmentFileEvent

import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_file', type=str, help='File to load')
parser.add_argument('--event_file', type=str, help='File to load')
# parser.add_argument('--mesh', type=str, help='Mesh to load')
parser.add_argument('--result_file', type=str, help="Where to save file")
args = parser.parse_args()

timestr = time.strftime("%Y%m%d-%H%M%S")
if args.result_file is None:
    npz_filename = os.path.splitext(args.label_file)[0]+("_voxel_results_%s.npz" % (timestr))
else:
    npz_filename = args.result_file

height = 256
width = 256
camera = ContinuousCamera(torch.eye(4), torch.eye(3), (width, height))

if "npz" in args.label_file:
    input_results = np.load(args.label_file)
    events = input_results['events']

    dist_coeffs = None
    fx = (25 / 36 * height)
    fy = (25 / 24 * height)
    cx = (height/2.0) - 0.5
    K = torch.tensor( [[fx, 0, cx],
                       [ 0,fx, cx],
                       [ 0, 0,  1]] )
elif "hdf5" in args.label_file:
    input_results = h5py.File(args.label_file, 'r')
    dist_coeffs = torch.tensor([-0.4088013193557094, 0.15476001540869858, -0.0015666113554986467,
    -0.0018004905208930813])
    fx = 1035.7900390250816 
    fy = 1036.2114996748978
    cx = 649.488661716263
    cy = 346.3857740654856
    K = torch.tensor( [[fx, 0, cx],
                       [ 0,fy, cy],
                       [ 0, 0,  1]] )
    event_f = h5py.File(args.event_file)

    print("computing labels")
    distances = input_results["event_contour_dist"][:]
    ev_dist_mask = input_results["event_distortion_mask"][:]
    labels = (distances < 10.) * (distances > 1.0) * (ev_dist_mask>0) # input_results["event_contour_labels"] # 

    distances = None

    print("truncating events")
    ex = event_f['x'][labels]
    ey = event_f['y'][labels]
    et = event_f['t'][labels]
    ep = event_f['p'][labels]
    print("stacking events")
    events = np.stack([ex,ey,et,ep], axis=-1)

    ex = None
    ey = None
    et = None
    ep = None

    # events = np.load("/media/ken/T71/MOEC_3D/hippo_events.npy")

    starting_time = events[0,2]
    events[:,2] -= starting_time
else:
    raise NotImplementedError()

# mesh = o3d.io.read_triangle_mesh(args.mesh)
# mesh.compute_vertex_normals()

camera.set_intrinsics(K, dist_coeffs)

poses = input_results['T_w_c'][:]

T_w_c = []
T_c_w = []

import trimesh

open3d_correction = trimesh.transformations.rotation_matrix(np.radians(180), [1,0,0])

open3d_correction = np.array( [[1,0,0,0],
                               [0,-1,0,0],
                               [0,0,-1,0],
                               [0,0,0,1]] )

for i in range(poses.shape[0]):
    cT_w_c = poses[i,...]
    T_w_c.append( cT_w_c )
    T_c_w.append( np.linalg.inv(cT_w_c) )


T_w_c = np.stack(T_w_c)
T_c_w = np.stack(T_c_w)

pose_coord_frames = []

for i in range(0,T_w_c.shape[0],20):
    cpf = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    cpf_vert = np.array(cpf.vertices).T
    cpf_vert = (T_w_c[i,:3,:3]@ cpf_vert)+T_w_c[i,:3,3,None]
    cpf.vertices = o3d.utility.Vector3dVector( cpf_vert.T )
    pose_coord_frames.append(cpf)

times = input_results['T_w_c_times'][:].astype(float) * 1e9
times -= starting_time
labels = np.ones( events.shape[0], dtype=int )

camera.set_trajectory(T_c_w, times)
camera.set_events(events, labels)

T = np.eye(4)
T[0,3] = -0.07
T[1,3] = 0.0
T[2,3] = 0.0

vol_size = 256

vol_config = [(vol_size, vol_size, vol_size), T, 0.10 / (vol_size // 2), "events"]
vol = Volume(*vol_config)

env = EnvironmentFileEvent(carve_dist=0.5, times_are_indexes=False)
env.set_camera(camera)
env.set_voxel_grid(vol)

env.compute_voxel_carving(100)
#env.compute_center_projection()

stats = env.voxel_grid.grid_stats()
print(stats)
#event_images = env.compute_event_images()

event_images = []

print(npz_filename)
np.savez_compressed(npz_filename, log_odds_grid=vol.log_odds_grid.cpu().numpy(), vol_config=vol_config, event_images=event_images)

#o3d_volume = vol.get_o3d_grid(0.1 + stats['mean'].item(), True)
#o3d.visualization.draw_geometries([o3d_volume, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)])
