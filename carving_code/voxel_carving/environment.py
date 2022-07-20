import torch
import numpy as np

from .utils import _device, genTimestampImage

import matplotlib.pyplot as plt

from tqdm import tqdm

import pdb

class Environment:
    def __init__(self):
        self.camera = None
        self.voxel_grid = None
        self.carve_distance = 1.0

    def set_camera(self, camera):
        self.camera = camera

    def set_voxel_grid(self, vg):
        self.voxel_grid = vg

    def compute_voxel_carving(self):
        rays = self.get_all_rays()
        self.voxel_grid.add(-20.0)

        for origin, direction in rays:
            self.voxel_grid.carve_ray(origin, direction,
                    value=0.5,
                    distance=self.carve_distance
                    )

        return


class EnvironmentSim(Environment):
    def __init__(self):
        super().__init__()
        self.carve_distance = 4.0

    def get_all_images(self):
        images = []
        for i in range(len(self.camera.poses)):
            self.camera.set_view(i)
            image = self.camera.compute_image()
            images.append(image)
        return images

    def get_all_mask_rays(self, use_n_views=None):
        data = []
        if use_n_views is None:
            dec = 1
        else:
            dec = len(self.camera.poses) // use_n_views

        for i in range(0, len(self.camera.poses), dec):
            self.camera.set_view(i)
            rays = self.camera.get_mask_rays()
            data.append(rays)
        return data

    def get_all_contour_rays(self):
        data = []
        for i in range(len(self.camera.poses)):
            self.camera.set_view(i)
            rays = self.camera.get_contour_rays()
            data.append(rays)
        return data

    def compute_mask_carving_projection(self):
        self.voxel_grid.save_metric_pts()
        self.voxel_grid.add(-10.0)
        rays = self.get_all_mask_rays()

        for origin, Ds in tqdm(rays):
            self.voxel_grid.carve_ray(origin, Ds, 0.01, only_unique=False, distance=self.carve_distance)
        return

    def compute_mask_carving(self, use_n_views=None):
        self.voxel_grid.add(-10.0)
        rays = self.get_all_mask_rays(use_n_views)

        for origin, Ds in tqdm(rays):
            self.voxel_grid.carve_ray(origin, Ds, 0.01, only_unique=False, distance=self.carve_distance)
        return

    def compute_contour_carving(self):
        self.voxel_grid.add(-10.0)
        rays = self.get_all_contour_rays()

        for origin, Ds in tqdm(rays):
            self.voxel_grid.carve_ray(origin, Ds, 0.01, only_unique=False, distance=self.carve_distance)
        return

    def compute_event_voxel_carving(self, parallel_level=100):
        import pdb
        self.voxel_grid.add(-10.0)

        events, poses = self.camera.get_labeled_event_pose_pairs(max_events=-1,skip_events=1, times_are_indexes=False)
        origins, Ds = self.camera.get_rays_for_pixels_unique_T(events[:,0], events[:,1], poses)

        print("Carving Volume")
        for i in tqdm(range(0, events.shape[0], parallel_level)):
            c_o = origins[i:i+parallel_level]
            c_d = Ds[i:i+parallel_level]

            self.voxel_grid.carve_ray(c_o, c_d, 0.01, only_unique=False, distance=self.carve_distance)

    def compute_method_stats(self, use_n_views):
        stats = {
                 "event": {},
                 "mask_frame": {},
                 "contour_frame": {},
                }
        stats['event']['num_ray_ops'] = self.camera.event_labels.sum()

        rays = self.get_all_mask_rays(use_n_views)
        stats['mask_frame']['num_ray_ops'] = sum( [r[1].shape[0] for r in rays] )

        return stats

class EnvironmentFileEvent(Environment):
    def __init__(self, carve_dist=None, times_are_indexes=True):
        super().__init__()
        if carve_dist is not None:
            self.carve_distance = carve_dist
        self.times_are_indexes = times_are_indexes

    def compute_event_images(self, num_events=1000):
        print("Generating Event Images")
        events, poses = self.camera.get_labeled_event_pose_pairs(max_events=-1,skip_events=10)
        event_image_list = []

        for i in tqdm(range(0, events.shape[0], num_events)):
            event_subset = events[i:i+num_events].cpu().numpy()
            event_images = genTimestampImage(event_subset)

            event_image_list.append(event_images[0] + event_images[1])

        return event_image_list

    def compute_center_projection(self):
        self.voxel_grid.add(-10.0)
        for p in self.camera.poses:
            origins, Ds = self.camera.get_center_rays(p)
            self.voxel_grid.carve_ray(origins, Ds, value=0.5, only_unique=False, distance=self.carve_distance)
        return

    def compute_voxel_carving(self, ec_size=10):
        self.voxel_grid.add(-10.0)

        events, poses = self.camera.get_labeled_event_pose_pairs(max_events=-1,skip_events=1, times_are_indexes=self.times_are_indexes)
        #events, poses = self.camera.get_labeled_event_pose_pairs()

        print("Carving Volume")
        for i in tqdm(range(0, events.shape[0], ec_size)):
            origins, Ds = self.camera.get_rays_for_pixels_unique_T(events[i:i+ec_size,0], events[i:i+ec_size,1], poses[i:i+ec_size])

            c_o = origins
            c_d = Ds

            self.voxel_grid.carve_ray(c_o, c_d, 0.01, only_unique=False, distance=self.carve_distance)

        return

class EnvironmentFileMask(Environment):
    def __init__(self):
        super().__init__()

    def compute_voxel_carving(self):
        self.voxel_grid.add(-10)
        for loc_i in range(len(self.camera.poses)):
            self.camera.set_view(loc_i)

            origin, direction = self.camera.get_mask_rays()

            if origin is None:
                continue

            self.voxel_grid.carve_ray(origin, direction, 0.01)
        return
