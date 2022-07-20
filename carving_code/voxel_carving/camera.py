import torch
import open3d as o3d
import numpy as np
import cv2

from .utils import _device, normalize, genTimestampImage
from .geometry import trace_ray_depth, trace_ray_color, invert_transform

import pdb
import matplotlib.pyplot as plt

class Camera:
    def __init__(self, T, K, sensor_size):
        self.T = T.to(_device)
        self.set_intrinsics(K)
        self.dist_coeffs = None

        self.w = sensor_size[0]
        self.h = sensor_size[1]

        self.events = None
        self.event_labels = None

        self.current_view_idx = None
        self.times = []
        self.poses = []

    def set_masks(self, masks):
        self.masks = torch.tensor(masks).to(_device)

    def get_labeled_event_pose_pairs(self, value=True, max_events=None, skip_events=None, times_are_indexes=False, start_e_idx=None, stop_e_idx=None):
        labeled_events = self.events[self.event_labels == value, :]
        event_inds = np.arange(0,len(self.events))[self.event_labels == value]

        if max_events is not None:
            labeled_events = labeled_events[:max_events:skip_events]
            event_inds = event_inds[:max_events:skip_events]

        if start_e_idx is not None and stop_e_idx is not None:
            labeled_events = labeled_events[start_e_idx:stop_e_idx:skip_events]
            event_inds = event_inds[start_e_idx:stop_e_idx:skip_events]

        if times_are_indexes:
            poses = self.get_poses_for_times( event_inds, "lin" )
        else:
            poses = self.get_poses_for_times( self.events[event_inds,2], "lin" )

        return torch.tensor(labeled_events).float().to(_device), torch.tensor(poses).float().to(_device)

    def set_masks(self, masks):
        self.masks = torch.tensor(self.masks).to(_device).float()

    def set_events(self, events, labels=None):
        self.events = events
        self.events = self.events.astype(float)
        self.events[:,2] -= (self.events[0,2] - self.times[0])

        self.events[:,:2] = self.undistort_points(self.events[:,:2])

        self.events = torch.tensor(self.events).to(_device).float()

        if labels is not None:
            self.event_labels = torch.tensor(labels).to(_device)
        else:
            self.event_labels = labels

    def compute_event_images(self, num_events=10000):
        assert self.events is not None

        if self.event_labels is not None:
            events = self.events[self.event_labels,:]
        else:
            events = self.events

        event_image_list = []

        for i in range(0, events.shape[0], num_events):
            event_subset = events[i:i+num_events].cpu().numpy()
            event_images = genTimestampImage(event_subset, _image_height=self.h, _image_width=self.w)

            ei = event_images[0] + event_images[1]

            ei = ei - ei.min()
            ei = ei / ei.max()

            event_image_list.append(ei)

        return event_image_list

    def set_view(self, i):
        if i >= len(self.times):
            raise ValueError()

        self.current_view_idx = i
        self.T = torch.tensor(self.poses[i]).to(_device).float()

    def set_intrinsics(self, K, dist_coeffs=None):
        if K.shape == (3,3):
            self.K = K.to(_device).float()
        else:
            self.K = torch.eye(3, device=_device)
            self.K[0,0] = K[0]
            self.K[1,1] = K[1]
            self.K[0,2] = K[2]
            self.K[1,2] = K[3]

        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs.to(_device).float()
        else:
            self.dist_coeffs = dist_coeffs

    def set_trajectory(self, poses, times):
        if type(poses) == torch.tensor:
            poses = [p.cpu().numpy() for p in poses]

        self.poses = np.stack([p for p in poses])
        self.times = times

    def undistort_points(self, d_pts):
        if self.dist_coeffs is None:
            return d_pts

        if len(d_pts.shape) == 2:
            d_pts = d_pts[:,None,:]

        if type(d_pts) is torch.tensor:
            d_pts = d_pts.cpu().numpy()

        u_pts = cv2.fisheye.undistortPoints(d_pts.astype(float), self.K.cpu().numpy(), self.dist_coeffs.cpu().numpy())
        u_pts = u_pts.squeeze()

        u_pts[:,0] = (self.K[0,0] * u_pts[:,0]) + self.K[0,2]
        u_pts[:,1] = (self.K[1,1] * u_pts[:,1]) + self.K[1,2]

        if type(d_pts) is torch.tensor:
            return torch.tensor(u_pts).to(_device).float()
        else:
            return u_pts

    def get_poses_for_times(self, times, method="lin"):
        idx2 = np.searchsorted(self.times, times)
        idx2[idx2 == self.times.shape[0]] = self.times.shape[0]-1
        T2 = self.poses[idx2]

        if method == "nn":
            return T2

        idx1 = idx2 - 1
        idx1[ idx1 < 0 ] = 0

        T1 = self.poses[idx1]

        t1 = self.times[idx1]
        t2 = self.times[idx2]

        t1[ t1 == t2 ] = times[ t1==t2 ] - 1

        dt = ((times - t1) / (t2-t1))[:,None,None]

        if method == "lin":
            Ti = dt*(T2-T1) + T1
            return Ti

        if method == "slerp":

            import pytorch3d.transforms
            T1 = torch.tensor(T1)
            T2 = torch.tensor(T2)

            dt = torch.tensor(dt)

            Ti = dt*(T2-T1) + T1

            q1 = pytorch3d.transforms.matrix_to_quaternion( T1[:,:3,:3] )
            q2 = pytorch3d.transforms.matrix_to_quaternion( T2[:,:3,:3] )

            # dot product
            d = torch.bmm( q1.view(q1.shape[0], 1, q1.shape[1]), q1.view(q1.shape[0], q1.shape[1], 1) ).squeeze()

            q1[ d<0, : ] *= -1.0
            d[ d<0 ] *= -1.0

            angle = torch.acos(d)

            isin = 1.0 / torch.sin(angle)

            q1 = q1 * (torch.sin((1.0-dt.squeeze()) * angle) * isin)[:,None]
            q2 = q2 * (torch.sin(dt.squeeze()*angle)*isin)[:,None]

            Ti[:,:3,:3] = pytorch3d.transforms.quaternion_to_matrix( q1 + q2 )

            return Ti.cpu().numpy()

    def get_rays_for_pixels(self, xv, yv):
        # The origin of our rays will be the optical center of the camera
        origin = invert_transform(self.T)[:3,3]

        Qs = torch.stack( [xv, yv, torch.ones_like(xv) ], dim=-1 )

        # is this the same as R^-1 @ K^-1 @ Qs  ????
        Qs = Qs @ (torch.inverse(self.K).t() @ torch.inverse(self.T[:3,:3]).t())

        Ds = normalize(Qs)

        return origin, Ds

    def get_rays_for_pixels_unique_T(self, xv, yv, Tv):
        TvT = torch.transpose(Tv,1,2)
        origins = torch.bmm(TvT[:,:3,:3], (-1*Tv[:,:3,3,None]))

        Qs_a = torch.stack( [xv, yv, torch.ones_like(xv) ], dim=-1 )
        Qs_b = (Qs_a @ torch.inverse(self.K).t())[:,:,None]
        Qs_c = torch.bmm(TvT[:,:3,:3], Qs_b)

        Ds = normalize(Qs_c.squeeze(-1))

        return origins.squeeze(-1), Ds

    def get_pixel_rays(self, flatten=False):
        # The direction is the R^-1 * K^-1 * [u;v;1]
        yv, xv = torch.meshgrid( [
            torch.arange(0, self.h, device=_device),
            torch.arange(0, self.w, device=_device),
            ] )

        xv = xv.float()
        yv = yv.float()

        origin, Ds = self.get_rays_for_pixels(xv, yv)

        if flatten:
            return origin, Ds.reshape(-1,3)
        else:
            return origin, Ds

class ContinuousCamera(Camera):
    def __init__(self, T, K, sensor_size):
        super().__init__(T,K,sensor_size)

        self.events = None

    def get_center_rays(self, pose):
        self.T = torch.tensor(pose).float()

        yv = torch.tensor( [ self.h//2 ] ).float()
        xv = torch.tensor( [ self.w//2 ] ).float()

        return self.get_rays_for_pixels_unique_T( xv, yv, self.T[None,...] )

class PreRenderedCamera(Camera):
    def __init__(self, T, K, sensor_size):
        super().__init__(T, K, sensor_size)

    def compute_contour(self):
        kernel = np.ones((5,5),np.uint8)

        mask = self.masks[self.current_view_idx]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        dialated_mask = cv2.dilate(mask, kernel, iterations=2)
        contour = dialated_mask - mask

        return torch.tensor(contour).long().to(_device) > 0.5

    def get_mask_rays(self):
        mask = torch.tensor(self.masks[self.current_view_idx]) < 0.5
        mask = mask[:,:,None].expand(-1, -1, 3)

        origin, Ds = self.get_pixel_rays()
        Ds = Ds[mask].reshape(-1,3)
        return origin, Ds

    def get_object_center_rays(self):
        self.K = torch.tensor(self.intrinsic).float()
        self.T = torch.tensor(self.poses[self.current_view_idx]).float()

        events = self.event_set()
        if events is None or events.numel() == 0:
            return None, None

        origin, Ds = self.get_pixel_rays()

        events_x = events[:,0]
        events_x = (events_x.max() - events_x.min())/2.0 + events_x.min()
        event_x = events_x.long()

        events_y = events[:,1]
        events_y = (events_y.max() - events_y.min())/2.0 + events_y.min()
        event_y = events_y.long()

        Ds = Ds[event_x:event_x+2,event_y:event_y+2,:].reshape(-1,3)
        return origin, Ds

    def get_center_rays(self):
        self.K = torch.tensor(self.intrinsic).float()
        self.T = torch.tensor(self.poses[self.current_view_idx]).float()

        origin, Ds = self.get_pixel_rays()
        Ds = Ds[self.h//2:self.h//2+2,self.w//2:self.w//2+2,:].reshape(-1,3)
        return origin, Ds

    def get_contour_rays(self):
        mask = self.compute_contour()

        mask = mask[:,:,None].expand(-1, -1, 3)

        self.K = torch.tensor(self.intrinsic).float()
        self.T = torch.tensor(self.poses[self.current_view_idx]).float()

        origin, Ds = self.get_pixel_rays()
        Ds = Ds[mask].reshape(-1,3)
        return origin, Ds

class SimCamera(Camera):
    def __init__(self, T, K, sensor_size):
        super().__init__(T, K, sensor_size)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.w, height=self.h, visible=False)
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.array([0.,0.,0.])

    def set_view(self, i):
        super().set_view(i)

        ctr = self.vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = self.T.cpu().numpy()
        param.intrinsic.intrinsic_matrix = self.K.cpu().numpy()
        ctr.convert_from_pinhole_camera_parameters(param)
        self.needs_update = True

    def update_renderer(self):
        if self.needs_update:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.needs_update = False

    def add_geometry(self, mesh):
        self.vis.add_geometry(mesh)
        self.needs_update = True

    def compute_image(self):
        self.update_renderer()
        img = self.vis.capture_screen_float_buffer(False)
        return torch.tensor(np.array(img)).to(_device)

    def compute_mask(self):
        self.update_renderer()
        depth = self.vis.capture_depth_float_buffer(False)
        depth = np.array(depth)
        mask = (depth > 0.001).astype(np.uint8)
        return torch.tensor(mask).to(_device)

    def compute_contour(self, kernel_size=(3,3)):
        self.update_renderer()
        depth = self.vis.capture_depth_float_buffer(False)
        depth = np.array(depth)
        mask = (depth > 0.001).astype(np.uint8)

        kernel = np.ones(kernel_size,np.uint8)

        dialated_mask = cv2.dilate(mask, kernel, iterations=1)
        contour = dialated_mask - mask

        return torch.tensor(contour).to(_device)

    def compute_depth_map(self):
        self.update_renderer()
        depth_img = self.vis.capture_depth_float_buffer(False)
        return torch.tensor(np.array(depth_img)).to(_device)

    def get_mask_rays(self):
        mask = self.compute_depth_map() < 0.001
        mask = mask[:,:,None].expand(-1, -1, 3)
        origin, Ds = self.get_pixel_rays()
        Ds = Ds[mask].reshape(-1,3)
        return origin, Ds

    def get_center_rays(self):
        yv = torch.tensor( [ self.h//2 ] ).float()
        xv = torch.tensor( [ self.w//2 ] ).float()

        return self.get_rays_for_pixels_unique_T( xv, yv, self.T[None,...] )

    def get_contour_rays(self):
        mask = self.compute_contour()
        mask = mask[:,:,None].expand(-1, -1, 3)

        origin, Ds = self.get_pixel_rays()
        Ds = Ds[mask].reshape(-1,3)
        return origin, Ds

    def generate_all_from_views(self, compute_list=[]):
        data = []

        for i in range(len(self.poses)):
            self.set_view(i)
            if len(compute_list) > 1:
                view_data = []
                for cf in compute_list:
                    view_data.append(cf())
                data.append(view_data)
            else:
                data.append( compute_list[0]() )

        return data

    def generate_all_images(self):
        return self.generate_all_from_views([self.compute_image])

    def generate_all_contours(self):
        return self.generate_all_from_views([self.compute_contour])

    def generate_events_with_contour_labels(self):
        images_and_contours = self.generate_all_from_views([self.compute_image, self.compute_contour])

        folder, _ = self.save_images([i[0] for i in images_and_contours])
        print(folder)
        folder, _ = self.save_images([i[1] for i in images_and_contours])
        print(folder)

        events = self.generate_events( [i[0] for i in images_and_contours] )
        contour_stack = np.stack( [i[1] for i in images_and_contours] )

        event_to_keyframe_ind = np.searchsorted(self.times, events[:,2])

        contour_test_ind = np.stack([event_to_keyframe_ind, events[:,1], events[:,0]]).astype(int)

        at_contour = contour_stack[ (contour_test_ind[0], contour_test_ind[1], contour_test_ind[2]) ]
        leaving_contour = contour_stack[ (contour_test_ind[0]-1, contour_test_ind[1], contour_test_ind[2]) ]

        labels = ( at_contour + leaving_contour ) > 0

        return events, labels

    def save_images(self, images=None):
        image_filenames = []
        import tempfile
        import os
        folder = tempfile.mkdtemp()
        for i, I in enumerate(images):
            filename = os.path.join(folder, "%06d.png" % i)
            if type(I) == torch.Tensor:
                I = I.cpu().numpy()
            cv2.imwrite(filename, (I*255).astype(np.uint8))
            image_filenames.append(filename)

        return folder, image_filenames

    def generate_events(self, images=None):
        if images is None:
            images = self.generate_all_images()

        self.temp_image_folder, image_filenames = self.save_images(images)

        images = None

        Cp, Cn = 0.5, 0.5
        refractory_period = 1e-4
        log_eps = 1e-3
        use_log = True
        import esim_py
        esim = esim_py.EventSimulator(
                Cp,  # contrast thesholds for positive 
                Cn,  # and negative events
                refractory_period,  # minimum waiting period (in sec) before a pixel can trigger a new event
                log_eps,  # epsilon that is used to numerical stability within the logarithm
                use_log,  # wether or not to use log intensity
                )

        events = esim.generateFromStampedImageSequence(
                image_filenames,
                self.times,
                )
        
        return events
