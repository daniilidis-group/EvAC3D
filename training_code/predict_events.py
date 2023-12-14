import torch
import trainers
import h5py
import pdb
from datasets import gen_discretized_event_volume, normalize_event_volume
from datasets import EventSegHDF5 
import cv2
import torch.nn.functional as F
import os
import numpy as np
import pickle
from scripts import read_calibration
from tqdm import tqdm
import argparse 
from pathlib import Path

def angle_diff(angle_a, angle_b):
    diff = angle_b - angle_a
    return (diff + np.pi) % (np.pi * 2) - np.pi

def proprocess_events(events):

    events_x = events[:, 0]
    events_y = events[:, 1]
    events_t = events[:, 2]
    events_p = events[:, 3]

    events_t = (events_t - events_t.min()).astype(float) / 1e6
    events_p[events_p < 0.1] = -1
    events = np.stack((events_x, events_y, events_t, events_p), axis=1)

    # subtract out min to get delta time instead of absolute
    events[:,2] -= np.min(events[:,2])

    # normalize the timestamps
    events[:, 2] = (events[:, 2] - events[:, 2].min()) \
                    / (events[:, 2].max() - events[:, 2].min())
    return events
    
class ModelInference:
    def __init__(self, CKPT_PATH, _device="cuda"):
        self.hparams = torch.load(CKPT_PATH, map_location=_device)['hyper_parameters']
        print("Loaded from {}, to device: {}".format(CKPT_PATH, _device))
        self.model = trainers.Net.load_from_checkpoint(CKPT_PATH, 
                                        num_input_channels=20, map_location=_device).to(_device)
        self._device = _device

    def inference(self, prev_events, cur_events):
        self.model.eval()
        with torch.no_grad():
            prev_event_volume = \
                    gen_discretized_event_volume(torch.tensor(prev_events),
                                                [self.hparams['num_input_channels'], 480, 640])
            prev_event_volume = normalize_event_volume(prev_event_volume).float().to(self._device)
            prev_event_volume = F.interpolate(prev_event_volume.unsqueeze(0), 
                                (self.hparams['width'], self.hparams['height']))
            # run network
            samples = torch.tensor(cur_events[:, :2]).to(self._device).float()
            pred = self.model.contour_net(prev_event_volume, samples.unsqueeze(0))
            good_mask = F.sigmoid(pred[0, 0, :]) > .5
            return good_mask

def pred_mask_fn(h5_file, calib_folder, ckpt, pose_dict_path, save_folder, device):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get events and then predict mask
    object_name = Path(h5_file).name.replace(".h5", "")
    f = h5py.File(h5_file, 'r')

    PATH = ckpt
    model = ModelInference(PATH, _device=device)

    base_folder = save_folder
    cal_folder = calib_folder

    print("Reading cam file")
    yml_f = os.path.join(cal_folder, "camchain-recons.yaml")
    camera_params = read_calibration(yml_f)
    pose_dict_path = pose_dict_path

    assert(os.path.exists(yml_f) and os.path.exists(pose_dict_path))

    print("Reading pose file")
    with open(pose_dict_path, 'rb') as pk:
        pose_dict = pickle.load(pk)

    # data hparams
    hparams = torch.load(PATH, map_location=device)['hyper_parameters']
    num_prev_events = hparams['num_prev_events']
    num_cur_events = hparams['num_cur_events']
    

    print("Finding start")
    # find the first inferenceable point
    begin_index = 0
    while (begin_index < num_prev_events):
        begin_index += 100
    seen_events = []
    seen_labels = []
    num_batches = (f['x'].shape[0] - begin_index) // num_cur_events

    print("Start processing")
    for i in tqdm(range(num_batches)):

        prev_end = begin_index + i*num_cur_events
        prev_start = np.maximum(prev_end - num_prev_events, 0)

        # prev
        x_prev = np.array(f['x'][prev_start:prev_end]).astype(float)
        y_prev = np.array(f['y'][prev_start:prev_end]).astype(float)
        t_prev = np.array(f['t'][prev_start:prev_end])
        p_prev = np.array(f['p'][prev_start:prev_end]).astype(float)
        p_prev[p_prev < 0.1] = -1
        prev_events = np.stack((x_prev, y_prev, t_prev, p_prev), axis=1).astype(np.float)
        prev_events = proprocess_events(prev_events)

        # get cur events
        cur_start = prev_end
        cur_end = cur_start + num_cur_events

        x_cur = np.array(f['x'][cur_start:cur_end]).astype(float)
        y_cur = np.array(f['y'][cur_start:cur_end]).astype(float)
        t_cur = np.array(f['t'][cur_start:cur_end])
        p_cur = np.array(f['p'][cur_start:cur_end])

        cur_events = np.stack((x_cur, y_cur, t_cur, p_cur), axis=1).astype(np.float)
        cur_events = proprocess_events(cur_events)
        raw_events = np.copy(cur_events)

        samples = np.stack((x_cur, y_cur), axis=1)

        half_width = 320
        half_height = 240
        samples[:, 0] = (samples[:, 0] - half_width) / half_width
        samples[:, 1] = (samples[:, 1] - half_height) / half_height

        event_labels = model.inference(prev_events, samples)
        seen_events.append(raw_events)
        seen_labels.append(event_labels.cpu().numpy())

        # map good events into image coord
        good_events = cur_events[event_labels.cpu().numpy(), :]
        cur_event_volume = \
                gen_discretized_event_volume(torch.from_numpy(good_events).float(),
                                    [hparams['num_input_channels'], 480, 640])
        cur_event_volume = normalize_event_volume(cur_event_volume).float().to(device)

        # inference size
        event_image = cur_event_volume.sum(0).cpu().numpy()
        event_image = (event_image - event_image.min()) \
                    / (event_image.max() - event_image.min())

        '''
        cv2.imshow("event_image", event_image)
        cv2.waitKey(1)
        '''

    seen_events = np.concatenate(seen_events, axis=0)
    seen_labels = np.concatenate(seen_labels, axis=0)

    # for all of the angels, find the best frames
    def angle_diff(angle_a, angle_b):
        diff = angle_b - angle_a
        return (diff + np.pi) % (np.pi * 2) - np.pi

    # find the closest match in 
    all_angles = np.array(f['pose_to_angle'])
    keys = np.array(list(pose_dict.keys()))
    all_poses = []
    for i in range(all_angles.shape[0]):
        diff = np.abs(angle_diff(keys, all_angles[i]))
        idx = np.argmin(diff)
        all_poses.append(pose_dict[keys[idx]])
    all_poses = np.array(all_poses)

    pose_to_event = np.array(f['pose_to_event']) - begin_index
    keep = pose_to_event > 0

    # only keep events between the start and the end pose
    seen_events = seen_events[:pose_to_event[-1], ...]
    seen_labels = seen_labels[:pose_to_event[-1]]

    full_dict = {
                "events": seen_events,
                "labels": seen_labels,
                "poses": pose_to_event[keep],
                "camera_Ts": all_poses[keep],
                "intrinsics": camera_params["intrinsics_event"]
                }
    # save
    output_file = os.path.join(base_folder, "all_info_{}.pkl".format(object_name))
    with open(output_file, 'wb') as f:
        pickle.dump(full_dict, f)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", type=str)
    parser.add_argument("calib_folder", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("pose_dict_path", type=str)
    parser.add_argument("--save_folder", type=str, default="output")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    pred_mask_fn(args.h5_file, args.calib_folder, args.ckpt, 
            args.pose_dict_path, args.save_folder, args.device)

