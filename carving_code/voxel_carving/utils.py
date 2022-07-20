import torch
import numpy as np
import open3d as o3d
import pdb

# Manage a default device for all of the simulation
#_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_device = 'cpu'
INF = torch.tensor([float('inf')], device=_device)

def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0., 0., 2.]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def center_mesh(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices)
    return model

def get_figure_image(o3d_obj, view_config, vis=None):
    vis = None

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1000, height=1000, visible=False)
        used_old_vis = False
    else:
        used_old_vis = True
        vis.clear_geometries()

    if type(o3d_obj) == list:
        center = o3d_obj[0].get_center()
    else:
        center = o3d_obj.get_center()

    if type(o3d_obj) == list:
        for oo in o3d_obj:
            vis.add_geometry(oo, not used_old_vis)
    else:
        vis.add_geometry(o3d_obj, not used_old_vis)

    ctr = vis.get_view_control()
    if 'X' not in view_config and not used_old_vis:
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = np.array(view_config['T'])
        param.intrinsic.intrinsic_matrix = np.array(view_config['K'])
        ctr.convert_from_pinhole_camera_parameters(param)
    elif not used_old_vis:
        view_dict = view_config['X']['trajectory'][0]
        ctr.change_field_of_view(view_dict['field_of_view'])
        ctr.set_zoom(view_dict['zoom'])
        ctr.set_front( np.array(view_dict['front'])[:,None] )
        ctr.set_up( np.array(view_dict['up'])[:,None] )
        ctr.set_lookat( np.array(view_dict['lookat'])[:,None] )

    param = ctr.convert_to_pinhole_camera_parameters()

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    # Convert to int based and convert from RGB to BGR
    return (np.array(img)[:,:,::-1] * 255).astype(np.uint8), vis


def preprocess_mesh(model, scale=None, set_base_zero=False):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    if set_base_zero:
        center[2] = min_bound[2]

    if scale is None:
        scale = np.max(max_bound - min_bound) / 2.0

    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model

def set_default_device(device):
    _device = torch.device(device)
    INF = torch.tensor([float('inf')], device=_device)

def default_device():
    return _device

def get_inf():
    return INF

def normalize(x):
    return x / torch.norm(x,dim=-1,keepdim=True)

def normalized_clip(x):
    x_norm = torch.norm(x,dim=-1,keepdim=True)
    return torch.where( x_norm<1.0, x, x / x_norm)

def for_every(array, func, *args, cat_dim=-1, function_of_object=False):
    buf = []

    for a in array:
        if function_of_object:
            buf.append(getattr(a, func)(*args))
        else:
            buf.append(func(*args, a))

    return torch.cat(buf, dim=cat_dim)

def calc_floor_ceil(events_coord) :
    coord_floor = np.floor(events_coord)
    coord_ceil = np.ceil(events_coord)
    coord_ceil[(coord_ceil - coord_floor) <= 0.01] += 1
    weight_floor = coord_ceil - events_coord
    weight_ceil = events_coord - coord_floor
    return ((coord_floor, weight_floor), (coord_ceil, weight_ceil))
    
def genTimestampImage(events, _thresh=0.01,
                      _image_height=720, _image_width=1280,
                      _start_width=0, _start_height=0):
    if events.shape[1] == 4 and not events.shape[0] == 4:
        events = events.T

    timestamp_pos_image = np.zeros((_image_height, _image_width))
    timestamp_neg_image = np.zeros((_image_height, _image_width))
    eventcount_pos_image = np.zeros((_image_height, _image_width))
    eventcount_neg_image = np.zeros((_image_height, _image_width))
    valid = np.all(np.stack([events[0, :] >= _start_width,
        events[0, :] < _start_width + _image_width - 1,
        events[1, :] >= _start_height,
        events[1, :] < _start_height + _image_height - 1], axis = 0), axis = 0)
    events = events[:,valid]
    events[0, :] -= _start_width
    events[1, :] -= _start_height
    events[2, :] -= events[2, 0]
    events[3, :] -= 0.5
    events[3, :] *= 2.0

    num_events = events.shape[1]
    events_xs = calc_floor_ceil(events[0, :])
    events_ys = calc_floor_ceil(events[1, :])
    events_info = np.zeros((4, 4 * num_events))
    for i in range(2) :
        for j in range(2) :
            weight = events_xs[i][1] * events_ys[j][1]
            ind = np.stack((events_xs[i][0], events_ys[j][0]), axis = 0)
            timestamp = weight * events[3, :] * events[2, :]
            events_info[:, (i * 2 + j) * num_events : (i * 2 + j + 1) * num_events] = \
                    np.concatenate((ind, weight[np.newaxis, :], timestamp[np.newaxis, :]), axis = 0)
   
    events_pos_info = events_info[:, events_info[3, :] > 0]
    events_neg_info = events_info[:, events_info[3, :] < 0]

    events_pos_inds = np.asarray(list(zip(events_pos_info[1, :], events_pos_info[0, :])), dtype = np.int32)
    events_neg_inds = np.asarray(list(zip(events_neg_info[1, :], events_neg_info[0, :])), dtype = np.int32)

    try:
        np.add.at(timestamp_pos_image, [events_pos_inds[:, 0], events_pos_inds[:, 1]], events_pos_info[3, :])
        np.add.at(eventcount_pos_image, [events_pos_inds[:, 0], events_pos_inds[:, 1]], 1)
    except:
        timestamp_pos_image = np.zeros((_image_height, _image_width))
        eventcount_pos_image = np.zeros((_image_height, _image_width))

    try:
        np.add.at(timestamp_neg_image, [events_neg_inds[:, 0], events_neg_inds[:, 1]], -events_neg_info[3, :])
        np.add.at(eventcount_neg_image, [events_neg_inds[:, 0], events_neg_inds[:, 1]], 1)
    except:
        timestamp_neg_image = np.zeros((_image_height, _image_width))
        eventcount_neg_image = np.zeros((_image_height, _image_width))
    
    #print timestamp_image
    #print eventcount_pos_image
    
    return (timestamp_pos_image / (eventcount_pos_image + 1e-5), \
            timestamp_neg_image / (eventcount_neg_image + 1e-5))
