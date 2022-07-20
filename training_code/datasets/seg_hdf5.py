import numpy as np
import h5py
import copy
import torch
import pdb
import os
import cv2
from .event_utils import gen_discretized_event_volume, normalize_event_volume
from scipy.spatial.transform import Rotation as R
from pathlib import Path, PurePath
import torchvision.transforms.functional as TF
from pykdtree.kdtree import KDTree
from utils import get_contour, draw_contours
import torch.nn.functional as F

class EventSegHDF5(torch.utils.data.Dataset):
    def __init__(self, path,  
                        width=640, 
                        height=480, 
                        num_input_channels=20,
                        num_samples=3000,
                        num_prev_events=200000,
                        num_cur_events=200000,
                        pixel_width=0.002,
                        pos_ratio=0.5,
                        sample_pixels=False, 
                        delta_t=-1,
                        use_new_ace=False,
                        image_to_mask=False,
                        image_to_contour=False,
                        dense_contour=False,
                        sdf=False,
                        max_length=-1,
                        dense_dist=False,
                        evaluation=False,
                        transform=None):
        self.path = path
        self.width = width
        self.height = height
        self.num_input_channels = num_input_channels
        self.loaded = False
        self.transform = transform
        self.num_samples = num_samples
        self.num_prev_events = num_prev_events
        self.num_cur_events = num_cur_events
        self.sample_pixels = sample_pixels
        self.pixel_width = pixel_width
        self.pos_ratio = pos_ratio
        self.sdf = sdf
        self.dense_dist = dense_dist
        self.max_length = max_length if max_length > 0 else np.inf
        self.use_delta_t = delta_t > 0
        self.delta_t = delta_t
        self.evaluation = evaluation
        self.use_new_ace = use_new_ace
        self.image_to_mask = image_to_mask
        self.image_to_contour = image_to_contour
        self.dense_contour = dense_contour

    def query_image(self, image_dir, mask_files):
        # load all image files
        ts = []
        image_names = []
        for f in os.listdir(image_dir):
            ts.append(int(f.replace(".png", "")))
            image_names.append(os.path.join(image_dir, f))
        ts = np.array(ts)
        image_names = np.array(image_names)

        mask_ts = np.array([int(f.replace(".png", "")) for f in mask_files])
        
        kd_tree = KDTree(np.array(ts))
        dist, idx = kd_tree.query(mask_ts, k=1)
        image_names = image_names[idx]

        return image_names

    def load(self):
        self.file = h5py.File(self.path, 'r')
        path = PurePath(self.path)

        filename = path.name.replace(".h5", "")
        self.t = self.file['t']

        mask_folder = str(path.parent) + "_frames_masks_labeled"
        image_folder = str(path.parent) + "_frames"

        self.mask_dir = PurePath(PurePath(mask_folder), filename)
        self.image_folder = PurePath(PurePath(image_folder), filename)

        self.mask_files = []
        names = []
        for f in os.listdir(self.mask_dir):
            if not f.endswith(".png"):
                continue

            ts = int(f.replace(".png", ""))

            # hack to fix ts
            if ts < self.t[100]*1000 or ts > self.t[-100]*1000:
                continue

            mask_f = os.path.join(self.mask_dir, f)
            image_f = os.path.join(self.image_folder, f)
            self.mask_files.append(mask_f)
            names.append(f)

        self.mask_files = np.array(self.mask_files)
        if self.image_to_contour or self.image_to_mask:
            self.image_files = self.query_image(self.image_folder, names)

        # events
        self.x = self.file['x']
        self.y = self.file['y']
        self.p = self.file['p']
        self.t = self.file['t']

        self.mask_ts = []
        self.binary_masks = []

        for ff in self.mask_files:
            ts = int(ff.split('/')[-1].replace('.png', ''))
            self.mask_ts.append(ts)

        order = np.argsort(self.mask_ts)
        self.mask_ts = np.array(self.mask_ts)
        self.mask_ts = self.mask_ts[order]
        self.mask_files = self.mask_files[order]
        if self.image_to_contour or self.image_to_mask:
            self.image_files = self.image_files[order]

        # match ts based on ts, self.t is in micro second,
        # mask_ts is in nanosecond
        kd_tree = KDTree(np.array(self.t))
        dist, idx = kd_tree.query(self.mask_ts/1000, k=1)
        self.mask_to_event = idx

        self.num_events = self.t.shape[0]
        self.loaded = True

    def close(self):

        self.events = None
        self.num_events = None
        self.file.close()
        self.length = None
        self.loaded = False
        self.binary_masks = None
        self.mask_ts = None

    def __len__(self):
        if not self.loaded:
            self.load()
        length = copy.deepcopy(len(self.mask_files))
        self.close()
        return length

    def __getitem__(self, idx):

        if not self.loaded:
            self.load()

        data = {}
        half_size = np.array([[640, 480]]) / 2

        if self.use_delta_t:
            prev_end_index = int(self.mask_to_event[idx])
            prev_events = self.get_delt_t_events(prev_end_index, delta_t=-1*self.delta_t)
        else:
            # get prev_events (only used for volumes)
            num_events = self.num_prev_events
            prev_end_index = int(self.mask_to_event[idx])
            prev_start_index = np.maximum(prev_end_index-num_events, 0)
            prev_events = self.get_events_between(prev_start_index, prev_end_index)

	# generate event_volume
        prev_event_volume = gen_discretized_event_volume(torch.from_numpy(prev_events).float(),
                                            [self.num_input_channels, 480, 640])
        prev_event_volume = normalize_event_volume(prev_event_volume).float()
        data["event_volume"] = prev_event_volume

        # normalize prev events
        prev_scaled_xy = (prev_events[:, :2] - half_size) / half_size
        prev_events[:, 2] = (prev_events[:, 2] - prev_events[0, 2]) / \
                                        (prev_events[-1, 2] - prev_events[0, 2])
        prev_scaled_events = np.concatenate((prev_scaled_xy, prev_events[:, 2:]), axis=-1)

        # padding if not enough
        if not self.use_delta_t:
            if prev_scaled_events.shape[0] < self.num_prev_events:
                missing = self.num_prev_events - prev_scaled_events.shape[0]
                sampled = np.random.randint(prev_scaled_events.shape[0], size=missing)
                prev_scaled_events = np.concatenate((prev_scaled_events, \
                                                prev_scaled_events[sampled, :]), axis=0)
            if prev_scaled_events.shape[0] < self.num_prev_events:
                print("Wrong number of events!!!!")
    
        # get current event samples
        cur_end_idx = np.minimum(prev_end_index+self.num_cur_events, self.t.shape[0])
        cur_start_idx = prev_end_index
        cur_events = self.get_events_between(cur_start_idx, cur_end_idx)

        data["start"] = cur_start_idx
        data["end"] = cur_end_idx

        # get mask
        ff = self.mask_files[idx]
        try:
            mask = cv2.resize(cv2.imread(ff, cv2.IMREAD_GRAYSCALE), (640, 480))
        except:
            print("Error file: ",  ff)

        if self.image_to_mask or self.image_to_contour:
            img_ff = self.image_files[idx]
            try:
                recons_image = cv2.resize(cv2.imread(img_ff, 
                                    cv2.IMREAD_GRAYSCALE), (640, 480))
                recons_image = recons_image.astype(np.float32) / 255.
                data['image'] = recons_image
            except:
                print("Error file: ",  ff)

        if self.dense_contour:
            kernel = np.ones((5, 5), 'uint8')
            contour_image = (255*((cv2.dilate(mask, kernel) - mask) > 0)).astype(np.uint8)

            one_kernel = np.ones((3, 3), 'uint8')
            contour_image_dilate = cv2.dilate(contour_image, one_kernel) > 0
            data['contour_gt'] = torch.tensor(contour_image_dilate.astype(np.float32)).unsqueeze(0)

        # contours
        if not self.use_new_ace:
            mask = mask.astype(np.float) / 255.
            contours = get_contour(mask > .5)
            gt_samples = contours[0][:, 0, :].astype(np.int)
            contour_image = np.zeros([480, 640], dtype=np.bool)
            contour_image[gt_samples[:, 1], gt_samples[:, 0]] = 1
            data["contour_image"] = torch.tensor(contour_image.astype(np.float32)).unsqueeze(0)

            contour_mask = np.zeros_like(mask)
            contour_mask = cv2.drawContours(contour_mask, [contours[0]], 
                                                -1, (1.), thickness=cv2.FILLED)
            data["mask"] = torch.tensor(contour_mask.astype(np.float32)).unsqueeze(0)

        else:
            kernel = np.ones((5, 5), 'uint8')
            contour_image = (cv2.dilate(mask, kernel) - mask) > 0
            data["contour_image"] = \
                    np.array(torch.tensor(contour_image.astype(np.float32)).unsqueeze(0))
            mask = mask.astype(np.float) / 255.
            data["mask"] = torch.tensor(mask.astype(np.float32)).unsqueeze(0)



        if self.image_to_mask:
            return data

        # get binary labels
        x_grid, y_grid = np.meshgrid(np.arange(contour_image.shape[1]), 
                                            np.arange(contour_image.shape[0]))
        contour_pix = np.stack((x_grid[contour_image], y_grid[contour_image]), axis=1)

        # contour pixels need be scaled first
        contour_pix = (contour_pix - half_size) / half_size
        kd_tree = KDTree(contour_pix)

        num_pos_samples = int(self.num_samples * self.pos_ratio)
        num_neg_samples = self.num_samples - num_pos_samples

        if self.image_to_contour:
            self.sample_pixels = True

        if not self.sample_pixels:

            scaled_xy = (cur_events[:, :2] - half_size) / half_size
            scaled_events = np.concatenate((scaled_xy, cur_events[:, 2:]), axis=-1)
            threshold = self.pixel_width * 2

            dist, idx = kd_tree.query(scaled_events[:, :2], k=1)
            near = dist < threshold
            events_pos = scaled_events[near].copy()
            events_neg = scaled_events[~near].copy()

            pos_samples = np.random.randint(events_pos.shape[0], size=num_pos_samples)
            neg_samples = np.random.randint(events_neg.shape[0], size=num_neg_samples)
            
            events_pos = events_pos[pos_samples, :]
            events_neg = events_neg[neg_samples, :]

            # distance from events to boundary
            event_dist_pos = dist[near][pos_samples]
            event_dist_neg = dist[~near][neg_samples]
        else:
            pixels_x, pixels_y = np.meshgrid(np.arange(contour_image.shape[1]),
                                                np.arange(contour_image.shape[0]))
            pixels = np.stack((pixels_x.flatten(), pixels_y.flatten()), axis=1)

            half_size = np.array([[640, 480]]) / 2
            scaled_pixels = (pixels - half_size) / half_size
            threshold = self.pixel_width * 2
            dist, idx = kd_tree.query(scaled_pixels, k=1)

            near = dist < threshold
            events_pos = scaled_pixels[near] 

            mask = mask.astype(np.bool).flatten()
            events_neg_inside = scaled_pixels[~near & mask]
            events_neg_outside = scaled_pixels[~near & ~mask]

            neg_samples_inside = np.random.randint(events_neg_inside.shape[0], 
                                                size=num_pos_samples//2)
            neg_samples_outside = np.random.randint(events_neg_outside.shape[0], 
                                                size=num_pos_samples-neg_samples_inside.shape[0])


            pos_samples = np.random.randint(events_pos.shape[0], size=num_pos_samples)

            events_pos = events_pos[pos_samples, :].astype(np.float32)
            events_neg_inside = events_neg_inside[neg_samples_inside, :].astype(np.float32)
            events_neg_outside = events_neg_outside[neg_samples_outside, :].astype(np.float32)
            events_neg = np.concatenate((events_neg_inside, events_neg_outside), axis=0)

            # distance from events to boundary
            event_dist_pos = dist[near][pos_samples]

            event_dist_neg_inside = dist[~near & mask][neg_samples_inside]
            event_dist_neg_outside = dist[~near & ~mask][neg_samples_outside]

            if self.sdf:
                event_dist_neg_outside *= -1
            event_dist_neg = np.concatenate((event_dist_neg_inside, event_dist_neg_outside))

        rand_sample = np.random.randint(low=0, high=scaled_xy.shape[0], size=1000)
        data["raw_events"] = scaled_xy.astype(np.float32)[rand_sample, :]
        data["raw_labels"] = near[rand_sample]
        data["events"] = np.concatenate((events_pos, events_neg), axis=0).astype(np.float32)
        data["labels"] = np.concatenate((torch.ones([num_pos_samples, 1]), 
                                            torch.zeros([num_neg_samples, 1])), axis=0)
        data["num_pos_samples"] = num_pos_samples
        data["num_neg_samples"] = num_neg_samples

        data["gt_dist"] = np.concatenate((event_dist_pos, event_dist_neg), axis=0).astype(np.float32)

        if (cur_events[:, :2].min()) < 0:
            print("Bad events!!!!!!!!!!!!!!!!!!!!!!!")

        #data["raw_events"] = cur_events

        return data

    def preprocess_events(self, events):
        # No events in this time window
        if events.shape[0] == 0:
            pdb.set_trace()

        # subtract out min to get delta time instead of absolute
        events[:,2] -= np.min(events[:,2])

        # normalize the timestamps
        events[:, 2] = (events[:, 2] - events[:, 2].min()) \
                        / (events[:, 2].max() - events[:, 2].min())

        # convolution expects 4xN
        events = events.astype(np.float32)
        return events

    def get_events_between(self, start_ind, end_ind):
        if not self.loaded:
            self.load()
        #events = self.events[start_ind:end_ind,:]
        
        events_x = np.array(self.x[start_ind:end_ind])
        events_y = np.array(self.y[start_ind:end_ind])
        events_t = np.array(self.t[start_ind:end_ind])

        # sutract to avoid overflow
        events_t = (events_t - events_t.min()).astype(float) / 1e6
        events_p = np.array(self.p[start_ind:end_ind]).astype(int)

        # change to -1, 1 if originally 0, 1
        events_p[events_p < 0.1] = -1
        events = np.stack((events_x, events_y, events_t, events_p), axis=1)

        if events.shape[0] == 0:
            print(start_ind, end_ind)
            pdb.set_trace()
        events = self.preprocess_events(events)

        return events

    '''
    def get_delt_t_events(self, ref_idx, delta_t=0.01):

        if not self.loaded:
            self.load()

        if delta_t < 0:
            # find events coarsely
            cur = ref_idx
            target_time = self.t[ref_idx] / 1e6 + delta_t
            increment = -100000
            while True:
                if cur < 0:
                    cur = 0
                    break;
                if self.t[cur] / 1e6 < target_time:
                    break;
                cur += increment
            events_x = np.array(self.x[cur:ref_idx])
            events_y = np.array(self.y[cur:ref_idx])
            events_t = np.array(self.t[cur:ref_idx]) / 1e6
            events_p = np.array(self.p[cur:ref_idx])
            min_index = cur + np.argmin(np.abs(events_t - target_time))
            start = min_index
            end = ref_idx
        else:
            # find events coarsely
            cur = ref_idx
            target_time = self.t[start_ind] / 1e6 + delta_t
            increment = 100000
            while True:
                if cur > self.events.shape[0]:
                    cur = self.events.shape[0]
                    break;
                if self.t[cur] / 1e6 > target_time:
                    break;
                cur += increment
            events_x = np.array(self.x[ref_idx:cur])
            events_y = np.array(self.y[ref_idx:cur])
            events_t = np.array(self.t[ref_idx:cur]) / 1e6
            events_p = np.array(self.p[ref_idx:cur])
            min_index = ref_idx + np.argmin(np.abs(events_t - target_time))
            start = ref_idx
            end = min_index

        events = np.stack((self.x[start: end], 
                            self.y[start: end], 
                            self.t[start: end], 
                            self.p[start: end]), axis=1)

        if events.shape[0] == 0:
            print(start, end)
            pdb.set_trace()

        events = self.preprocess_events(events)
        return events
    '''


        

