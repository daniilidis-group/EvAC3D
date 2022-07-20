from o3d_voxel_example import open3d_tutorial as o3dtut
import open3d as o3d
import numpy as np
import os
import mcubes
import torch

import sys
import pdb

from voxel_carving import SimCamera, Volume, normalize, transform_rays, invert_transform, EnvironmentSim, preprocess_mesh, get_extrinsic, center_mesh, apply_global_velocity

import matplotlib.pyplot as plt
import esim_py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--shapenet_dir', type=str, help='ShapeNet base directory',  default="/home/ken/datasets/ShapeNetCore.v1")
parser.add_argument('--category', type=str, help='model to load', default="02691156")
parser.add_argument('--model_id', type=str, help='model to load', default="c88275e49bc23ee41af5817af570225e")
parser.add_argument('--mesh_file', type=str, help="Mesh filename override", default=None)
parser.add_argument('--use_n_views', type=int, help="Use n views for mask", default=None)
parser.add_argument('--method', type=str, help='Carving method')
parser.add_argument('--trajectory', type=str, help='Trajectory method', default="circle")
args = parser.parse_args()

if args.mesh_file is None:
    mesh_fn = os.path.join(args.shapenet_dir, args.category, args.model_id, "model.obj")
else:
    mesh_fn = args.mesh_file

mesh = o3d.io.read_triangle_mesh(mesh_fn)

mesh.compute_vertex_normals()
mesh = preprocess_mesh(mesh)
mesh.translate( np.array([[0.0],[0.],[0.1]]) )


if args.trajectory == "circle":
    T_start = np.eye(4)
    T_start[2,3] = 2.0
    camera_path = [T_start]
    
    for i in range(800):
        T_cur = camera_path[-1].copy()
        rpy = [0.01, 0.0, 0.0]
        camera_path.append(apply_global_velocity(T_cur, 0, 0, 0, rpy[0], rpy[1], rpy[2]))

#     circle_verts = [[np.sin(i),np.cos(i),0] for i in np.linspace(-np.pi,np.pi,800)]
#     camera_path = [get_extrinsic(xyz) for xyz in circle_verts]
elif args.trajectory == "const":
    T_start = np.eye(4)
    T_start[2,3] = 2.0
    camera_path = [T_start]
    
    xyz = [0.00, 0.00, 0.00]
    rpy = [0.00, np.pi/200,0.00]

    for i in range(400):
        if i % 200 == 0:
            rpy[0], rpy[1] = rpy[1], rpy[0]
            print(i, rpy)

        T_cur = camera_path[-1].copy()
        camera_path.append(apply_global_velocity(T_cur, xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]))

    camera_path = camera_path + list(reversed(camera_path))

elif args.trajectory == "rand":
    T_start = np.eye(4)
    T_start[2,3] = 2.1
    camera_path = [T_start]
    
    for i in range(800):
        T_cur = camera_path[-1].copy()
        rpy = np.random.rand(3) * 0.02
        rpy[2] = 0
        camera_path.append(apply_global_velocity(T_cur, 0, 0, 0, rpy[0], rpy[1], rpy[2]))
else:
    raise NotImplementedError()

T = torch.tensor(camera_path[0]).float()
K = torch.tensor( [250., 250., 400/2 - 0.5, 400/2 - 0.5] )

camera = SimCamera(T, K, (400, 400))

camera.add_geometry(mesh)
depth = camera.compute_depth_map()
img = camera.compute_image()

T = torch.eye(4)
T[0,3] = 1.2
T[1,3] = 1.2
T[2,3] = 1.2

vol_size = 256
vol_config = [(vol_size, vol_size, vol_size), T, 1.2 / 128, "shapenet"]
vol = Volume(*vol_config)

method_split = args.method.split("_")
if len(method_split) == 2:
    divider = int(method_split[1])
else:
    divider = 1

camera_path = np.stack(camera_path)
camera_times = np.linspace(0,1,camera_path.shape[0])
camera_times = (float(camera_path.shape[0]) / 800.)*np.linspace(0,1,camera_path.shape[0])

camera.set_trajectory(camera_path, camera_times)

env = EnvironmentSim()
env.set_camera(camera)
env.set_voxel_grid(vol)

event_images=[]

if "events" in args.method:
    events, labels = camera.generate_events_with_contour_labels()
    camera.set_events(events, labels)
    event_images = camera.compute_event_images()
    camera.save_images( event_images )

    o3d_volume = env.compute_event_voxel_carving()
elif "mask" in args.method:
    o3d_volume = env.compute_mask_carving(args.use_n_views-1)
elif "contour" in args.method:
    o3d_volume = env.compute_contour_carving()
    pass
elif "stats" in args.method:
    events, labels = camera.generate_events_with_contour_labels()
    camera.set_events(events, labels)
    print(env.compute_method_stats(args.use_n_views-1))
    sys.exit(0)
else:
    raise NotImplementedError()

if args.mesh_file is None:
    npz_filename = "results/%s_%s_%s_%s.npz" % (args.category, args.model_id, args.method, args.trajectory)
else:
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    npz_filename = "results/%s_%s_%s_%s" % (os.path.basename(args.mesh_file)[:-4], args.method, args.trajectory, str(args.use_n_views))

print(npz_filename)
np.savez_compressed(npz_filename, log_odds_grid=vol.log_odds_grid.cpu().numpy(), vol_config=vol_config, event_images=event_images)
