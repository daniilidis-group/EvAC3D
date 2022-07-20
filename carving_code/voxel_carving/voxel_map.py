import torch
import open3d as o3d
import numpy as np

from .geometry import transform_rays, invert_transform
from .utils import _device, normalize

import pdb
import cc3d

class Volume:
    def __init__(self, shape, T, voxel_size, carving_type="event"):
        self.log_odds_grid = torch.zeros(shape).to(_device)
        self.T = torch.tensor(T).float().to(_device)
        self.voxel_size = voxel_size
        self.grid_pts = None
        self.carving_type = carving_type

    def grid_stats(self):
        return {
            "min": self.log_odds_grid.min(),
            "max": self.log_odds_grid.max(),
            "mean": self.log_odds_grid.mean(),
               }

    def add(self, val):
        self.log_odds_grid += val
    
    def decay(vol, val):
        self.log_odds_grid *= val
    
    def ray_to_update(self, ray_origin, ray_direction):
        return

    def save_metric_pts(self):
        if self.grid_pts is None:
            x = torch.arange(0, self.log_odds_grid.shape[0])
            y = torch.arange(0, self.log_odds_grid.shape[1])
            z = torch.arange(0, self.log_odds_grid.shape[2])
            grid = torch.meshgrid(x,y,z)
            self.grid_pts = torch.stack(grid)

        pdb.set_trace()

    def get_object_center(self, height_offset=0, threshold=0.01, gt=True, manual_center=None):
        if manual_center is not None:
            return torch.tensor(manual_center)
        x = torch.arange(0, self.log_odds_grid.shape[0])
        y = torch.arange(0, self.log_odds_grid.shape[1])
        z = torch.arange(0, self.log_odds_grid.shape[2])
        grid = torch.meshgrid(x,y,z)
        probability_grid = self.get_probability_grid()

        if "event" in self.carving_type:
            grid_mask = probability_grid > probability_grid.min() + 0.1
        elif "contour" in self.carving_type:
            grid_mask = probability_grid > probability_grid.min() + 0.1
        elif "shapenet" in self.carving_type:
            grid_mask = probability_grid > probability_grid.min() + 0.1
        else:
            grid_mask = probability_grid < probability_grid.min() + 0.01

        pg_sum = probability_grid[grid_mask].sum()

        x_mean = (grid[0][grid_mask] * probability_grid[grid_mask]).sum() / pg_sum
        y_mean = (grid[1][grid_mask] * probability_grid[grid_mask]).sum() / pg_sum
        z_mean = (grid[2][grid_mask] * probability_grid[grid_mask]).sum() / pg_sum

        return torch.stack([x_mean, y_mean, z_mean+height_offset])

    def normalize_ray(self, ray_origin, ray_direction):
        nro = ray_origin
        nrd = ray_direction
        return nro, nrd

    def get_probability_grid(self):
        odds_grid = torch.exp(torch.clamp(self.log_odds_grid+7.5, -50, 50))
        return odds_grid / (1.+odds_grid)

    def get_connected_components(self, value=0.0, gt=True, size=(128,128,128), use_connected_components=False, height_offset=0, manual_center=None):
        cropped_grid = self.log_odds_grid.cpu().numpy() # [x_min:x_max, y_min:y_max, z_min:z_max]
        if gt:
            threshold = cropped_grid > value
        else:
            threshold = cropped_grid < value
        labels_out = cc3d.connected_components( threshold )

        xyz = self.get_object_center(height_offset, manual_center=manual_center).long()
        object_idx = labels_out[ xyz[0], xyz[1], xyz[2] ]

        return labels_out, object_idx


    def segment_object(self, value=0.0, gt=True, size=(128,128,128), use_connected_components=False, height_offset=0, manual_center=None):
        cropped_grid = self.log_odds_grid.cpu().numpy() # [x_min:x_max, y_min:y_max, z_min:z_max]
#         from scipy import signal
#         cropped_grid = signal.medfilt(cropped_grid, (3,3,3))
        if gt:
            threshold = cropped_grid > value
        else:
            threshold = cropped_grid < value

        if use_connected_components:
            xyz = self.get_object_center(height_offset, manual_center=manual_center).long()
            labels_out = cc3d.connected_components( threshold )
            stats = cc3d.statistics(labels_out)
            closest_center_idx = np.argmin( np.linalg.norm(stats['centroids'][1:] - xyz[None,:].cpu().numpy(), axis=-1) ) + 1

            object_idx = labels_out[ xyz[0], xyz[1], xyz[2] ]

            object_label = torch.tensor(labels_out == object_idx)
        else:
            object_label = torch.tensor(threshold)
        return object_label

    def get_npy_point_cloud(self, value=0.0, gt=True, size=(128,128,128), use_connected_components=False, height_offset=0, manual_center=None):
        obj_mask = self.segment_object(value, gt, size, use_connected_components, height_offset, manual_center=manual_center)
        pts = torch.stack(torch.where(obj_mask))

        return pts.T.float().cpu().numpy()

    def get_o3d_grid(self, value=0.0, gt=True, size=(128,128,128), use_connected_components=False, height_offset=0, manual_center=None):
        npy_pts = self.get_npy_point_cloud(value, gt, size, use_connected_components, height_offset, manual_center=manual_center)
        npy_pts *= self.voxel_size
        npy_pts = npy_pts - self.T[None,:3,3].cpu().numpy()

        if npy_pts.size == 0:
            return o3d.geometry.VoxelGrid()

        o3d_pts = o3d.geometry.PointCloud()
        o3d_pts.points = o3d.utility.Vector3dVector(npy_pts)

        import matplotlib.pyplot as plt
        cm = plt.get_cmap('plasma')

        normalized_height = npy_pts[:,2] - npy_pts[:,2].min()
        normalized_height /= normalized_height.max()

        o3d_pts.colors = o3d.utility.Vector3dVector(cm(normalized_height)[:,:3])

        o3d_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pts, self.voxel_size)
        return o3d_grid

    def get_mcubes_mesh(self, value=0.0, gt=True, size=(128,128,128), use_connected_components=False, height_offset=0, metric=True, manual_center=None):
        import mcubes
        binary_grid = self.segment_object(value, gt, size, use_connected_components, height_offset, manual_center=manual_center)
        binary_grid = np.pad(binary_grid.cpu().numpy(), ( (1,1), (1,1), (1,1) ), mode='constant')

        vertices, triangles = mcubes.marching_cubes(binary_grid, 0)

        mesh = o3d.geometry.TriangleMesh()
        if metric:
            vertices = vertices * self.voxel_size
            vertices = vertices - self.T[None,:3,3].cpu().numpy()

        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh

    def get_mcubes_mesh_real_values(self, value=0.0, gt=True, size=(128,128,128), use_connected_components=False, height_offset=0, metric=True):
        import mcubes
        binary_grid = self.segment_object(value, gt, size, use_connected_components, height_offset)
        binary_grid = np.pad(binary_grid.cpu().numpy(), ( (5,5), (5,5), (5,5) ), mode='constant')

        import scipy
        from skimage import exposure
        dilated_grid = scipy.ndimage.morphology.binary_dilation(binary_grid, np.ones((3,3,3)))

        lo = self.log_odds_grid.cpu().numpy()
        lo = (lo - lo.min()) / (lo.max() - lo.min())
        equalized_grid = exposure.equalize_adapthist( lo )

        voxel_grid = np.pad(equalized_grid, ( (5,5), (5,5), (5,5) ), mode='constant')
        voxel_grid[np.logical_not(dilated_grid)] = 1
        voxel_grid = voxel_grid.max() - voxel_grid

        vertices, triangles = mcubes.marching_cubes(voxel_grid, 0)

        mesh = o3d.geometry.TriangleMesh()
        if metric:
            vertices = vertices * self.voxel_size
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh

    def carve_projection(self, mask, camera):
        return

    def carve_ray(self, origin, direction, value=None, only_unique=False, distance=1.0, decay=False, ray_samples=1024):
        origin, direction = transform_rays(origin, direction, self.T)
        direction = normalize(direction)

        # Goal shape for everything is -- (ray_length, num_rays, 3)
        locations = torch.linspace(0, distance, ray_samples)[:,None,None]
        if len(origin.shape) == 1:
            origin = origin[None,None,:]
        else:
            origin = origin[None,:,:]
        direction = direction.view(-1,3)[None,:,:]


        voxel_pts = origin + (locations * direction)
        voxel_pts *= 1./self.voxel_size
        voxel_pts = torch.round(voxel_pts).long()

        ind_x = voxel_pts[:,:,0].ravel()
        ind_y = voxel_pts[:,:,1].ravel()
        ind_z = voxel_pts[:,:,2].ravel()

        vals = torch.zeros_like(voxel_pts[:,:,2]).float()
        if value is not None and decay:
            vals = vals + torch.linspace(value, 0, ray_samples)[:,None]
        elif value is not None:
            vals += value

        vals = vals.ravel()

        bounds_mask = ((ind_x >= 0) & (ind_x<self.log_odds_grid.shape[0]))\
                     &((ind_y >= 0) & (ind_y<self.log_odds_grid.shape[1]))\
                     &((ind_z >= 0) & (ind_z<self.log_odds_grid.shape[2]))

        ind_x = ind_x[bounds_mask]
        ind_y = ind_y[bounds_mask]
        ind_z = ind_z[bounds_mask]
        vals = vals[bounds_mask]

        if ind_x.numel() == 0:
            return

        if only_unique:
            inds = torch.stack((ind_x, ind_y, ind_z))
            inds = torch.unique(inds, dim=1)

            ind_x = inds[0,:]
            ind_y = inds[1,:]
            ind_z = inds[2,:]

        self.log_odds_grid.index_put_([ind_x, ind_y, ind_z], vals, accumulate=True)
