#from o3d_voxel_example import open3d_tutorial as o3dtut
import open3d as o3d
import numpy as np
import os
import mcubes
import torch

import pdb

from voxel_carving import ContinuousCamera, Volume, normalize, transform_rays, invert_transform, EnvironmentFileEvent

import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='File to load')
parser.add_argument('--original_frame', type=str, help='original camera frame', default="prophesee")
parser.add_argument('--calib', type=str, help='Calibration File', default="/home/ken/datasets/EventRecon/camchain-recons.yaml")
args = parser.parse_args()

timestr = time.strftime("%Y%m%d-%H%M%S")
npz_filename = os.path.splitext(args.file)[0]+("_voxel_results_%s.npz" % (timestr))

camera = ContinuousCamera(torch.eye(4), torch.eye(3), (1280,720))
with open(args.file, 'rb') as f:
    import pickle
    network_results = pickle.load(f)

print("events loaded")

with open(args.calib, 'r') as f:
    import ruamel.yaml as yaml
    calibration = yaml.load(f)

dist_coeffs = torch.tensor(calibration['cam1']['distortion_coeffs'])
K = torch.tensor(calibration['cam1']['intrinsics'])

camera.set_intrinsics(K, dist_coeffs)

poses = np.stack([np.linalg.inv(p) for p in network_results['camera_Ts']])

events = network_results['events']
times = network_results['poses']

camera.set_trajectory(poses, times)
labels = network_results['probability'] > 0.95

camera.set_events(events, labels)

T = np.eye(4)
T[0,3] = -0.07
T[1,3] = 0.0
T[2,3] = 0.0

vol_size = 256

vol_config = [(vol_size, vol_size, vol_size), T, 0.1 / (vol_size//2), "events"]
vol = Volume(*vol_config)

env = EnvironmentFileEvent(0.5, True)
env.set_camera(camera)
env.set_voxel_grid(vol)

print("Starting Carving")
env.compute_voxel_carving()
# env.compute_center_projection()

stats = env.voxel_grid.grid_stats()
print(stats)
# event_images = env.compute_event_images()

event_images = []
np.savez_compressed(npz_filename, log_odds_grid=vol.log_odds_grid.cpu().numpy(), vol_config=vol_config, event_images=event_images)

# o3d_volume = vol.get_o3d_grid(stats['mean'].item(), True)
# o3d.visualization.draw_geometries([o3d_volume])
