# EvAC3D From Event-based Apparent Contours to 3D Models via Continuous Visual Hulls

This is the official implementation of the paper. If you use **MOEC-3D** dataset or the code, please cite our work:

```bibtex
@inproceedings{wang2022evac3d,
  title={EvAC3D: From event-based apparent contours to 3d models via continuous visual hulls},
  author={Wang, Ziyun and Chaney, Kenneth and Daniilidis, Kostas},
  booktitle={European conference on computer vision},
  pages={284--299},
  year={2022},
  organization={Springer}
}
```
A copy of the paper can be found [here](datasets/seg_hdf5.py). For more information regarding the project, please visit our [project page](https://www.cis.upenn.edu/~ziyunw/evac3d/).

# MOEC-3D Dataset
We release ground truth object ply files, events, calibration files and camera poses [here](https://drive.google.com/drive/u/1/folders/1frNjCATu2I2KAcvfPMHctmLVeGTP9nmg). 

## How to use our data for training

The `learning` folder has the poses, events, masks that we used for training and testing. Here is what is in the `learning` folder. Folder `event_files` has preprocessed events. `gt_scans` contains all ground truth scanning. `masks` contains the object masks from the camera view (inside 1 and outside 0), which one can use to lable the Apparent Contour Events. The camera poses are obtained by looking up the current angle (`pose_to_angle` in the event h5 file). The camera pose is obtained by accessing the nearest angle key in the pose dictionary stored in `poses_objects.pkl`. `pose_to_event` gives the indices of each pose with respect to the event stream. Optionally, a slerp interpolation can be used to assign different poses to events through interpolation. Camera calibration files are given with `camchain-recons_*.yaml` files. This is obtained using Kalibr. For carving, we would need the intrinsics and distortion parameters to map events to rays in the metric world.

## How to use our data for carving

### Object carving with network prediction
After the network has been trained, we save them into an intermediate format for carving. An example script for running such inference is in `training_code/predict_events.py`. The script saves events, predicted labels, cameras poses, calibration and camera timestamps into a .pkl file. This file is then used as the argument to `--file` in the carving code.

To run the carving code, simply run:

`python --file [prediction pickle file] --calib [calibration file path]`

### Real-world animal carving
In [this folder](https://drive.google.com/drive/u/1/folders/1h2mvxZbQSwfjWQobNtU5eo3FagievBtS), we provide examples labels and configurations for running `carving_animals.py`. This script generates the same `.npz` file format as `carving_objects.py`, which can be used for mesh extraction.

### Evaluation
Given the carved volumes saved as npz files, You can run `evaluation.py` to convert soft occupancy to meshes. Some default paramters are given for two example objects.