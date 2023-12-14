import matplotlib.pyplot as plt
import numpy as np
import torch
import open3d as o3d
import torch
from pykdtree.kdtree import KDTree
from voxel_carving import Volume, doICP, get_figure_image
import ruamel.yaml as yaml
import open3d as o3d
from pytorch3d.structures import Meshes

from scipy import ndimage
from tqdm import tqdm

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], vmin=volume.min(), vmax=volume.max())
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title(str(ax.index))

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title(str(ax.index))

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

# expect [N, 3]
def compute_chamfer(source_cloud, target_cloud, 
                        source_normal, target_normal):

    source_cloud = source_cloud
    target_cloud = target_cloud

    tree_source = KDTree(source_cloud)
    dist_forward, idx_forward = tree_source.query(target_cloud, k=1)

    tree_target = KDTree(target_cloud)
    dist_backward, idx_backward = tree_target.query(source_cloud, k=1)

    cos_forward = np.abs(source_normal[idx_forward]*target_normal).sum(axis=-1)
    cos_backward = np.abs(source_normal*target_normal[idx_backward]).sum(axis=-1)

    output = {"chamfer": (dist_forward.mean() + dist_backward.mean()) / 2,
                "cos": (cos_backward.mean() + cos_forward.mean()) / 2}
    return output

def optimize_source_to_target(source_cloud, target_cloud, total_iters, source_cloud_faces=None, orig_mesh=None, lambdas=[50.0,0.1]):
    if total_iters == 0:
        return np.zeros_like( source_cloud )

    source_cloud = source_cloud.astype(np.float32)
    target_cloud = target_cloud.astype(np.float32)

    tree_target = KDTree(target_cloud.astype(source_cloud.dtype))

    if source_cloud_faces is not None:
        source_cloud_faces = torch.tensor(source_cloud_faces)

    source_cloud = torch.tensor(source_cloud)
    target_cloud = torch.tensor(target_cloud)

    source_offset = torch.zeros( source_cloud.shape, requires_grad=True )

    from torch.optim import SGD, Adam
    opt = SGD( [source_offset], lr=0.001 )

    dist_loss_lambda = lambdas[0]
    smoothing_lambda = lambdas[1]

    for i in tqdm(range(total_iters)):
        opt.zero_grad()

        source_shifted = source_cloud + source_offset

        _, idx = tree_target.query( source_shifted.detach().cpu().numpy() )
        idx = torch.tensor(idx.astype(np.int64))

        distance = torch.pow(source_shifted - target_cloud[idx], 2).sum(dim=1)
        dist_mask = distance < 1000.0 # Close objects

        if source_cloud_faces is not None:
            ss_mesh = Meshes(source_shifted[None,:,:], source_cloud_faces[None,:,:])
            vp = ss_mesh.verts_packed()

            with torch.no_grad():
                L = ss_mesh.laplacian_packed()

            smoothing_loss = L.mm(vp).norm(dim=1)

            sc = smoothing_lambda*smoothing_loss[ dist_mask ].sum()
            sf = smoothing_lambda*smoothing_loss[ ~dist_mask ].sum()

            loss = dist_loss_lambda * distance[ dist_mask ].sum()
            dist_loss_lambda = dist_loss_lambda * 0.9

            #(loss).backward()
            (loss + sc + sf).backward()
        else:
            distance = distance
            loss = distance.sum()
            loss.backward()

        opt.step()

        opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] * 0.99

        if orig_mesh is not None:
            opt_mesh = o3d.geometry.TriangleMesh()
            opt_mesh.triangles = orig_mesh.triangles
            opt_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(orig_mesh.vertices) + source_offset.detach().cpu().numpy())
            opt_mesh.compute_vertex_normals()
            opt_mesh.paint_uniform_color( np.array([0.5, 0.0, 0.5]) )

            o3d.visualization.draw_geometries([opt_mesh, orig_mesh])

    return source_offset.detach().cpu().numpy()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='File to load')
parser.add_argument('--mesh', type=str, help='Mesh to load')
parser.add_argument('--height_offset', type=int, help='Offset due to legs on animals', default=0)
parser.add_argument('--crop_top', type=int, help='100 elephant, 200 mustard', default=-1)
parser.add_argument('--crop_bottom', type=int, help='100 elephant, 200 mustard', default=0)
parser.add_argument('--debug', action='store_true', help='Debug visualizations')
parser.add_argument('--figure_config', type=str, help='View point configuration')
parser.add_argument('--mesh_config', type=str, help='GT mesh configuration')
parser.add_argument('--threshold', type=float, help='Segmentation Threshold', default=-9.99)
parser.add_argument('--manual_center', nargs=3, type=int, help='Manually set the center of object', default=None)
parser.add_argument('--exp', type=str, default="", help='GT mesh configuration')

parser.add_argument('--optimizer_iterations', type=int, help='Number of iterations for the optimizer', default=10)
parser.add_argument('--optimizer_params', type=float, nargs=2, help='Lambdas - chamfer, laplacian', default=[50.0, 0.1])

args = parser.parse_args()
saved_volume = np.load(args.file, allow_pickle=True)

vol_config = list(saved_volume['vol_config'])

vol = Volume(*vol_config)
volume = saved_volume['log_odds_grid'][:,:,args.crop_bottom:args.crop_top]
resolution = saved_volume['vol_config'][2]
shape = saved_volume['vol_config'][0]

volume = ndimage.uniform_filter(volume, 5)

vol.log_odds_grid = torch.tensor( volume )

print("Getting mesh")
pred_mesh = vol.get_mcubes_mesh(args.threshold, False, use_connected_components=True, height_offset=args.height_offset, manual_center=args.manual_center)
pred_mesh.compute_vertex_normals()

print("Getting o3d grid")
o3d_volume = vol.get_o3d_grid(args.threshold, False, use_connected_components=True, height_offset=args.height_offset, manual_center=args.manual_center)

print("Getting o3d shell")
shell_threshold = volume.mean() - volume.std()

o3d_shell = vol.get_o3d_grid(shell_threshold, True, use_connected_components=False, height_offset=args.height_offset, manual_center=args.manual_center)
npy_shell = vol.get_npy_point_cloud(shell_threshold, True, use_connected_components=False, height_offset=args.height_offset, manual_center=args.manual_center)
npy_shell *= vol.voxel_size
npy_shell = npy_shell - vol.T[None,:3,3].cpu().numpy()

print("Getting connected components volume")
cc_vol, selected_idx = vol.get_connected_components(args.threshold, False, use_connected_components=True, height_offset=args.height_offset, manual_center=args.manual_center)

print("Optimizing Mesh")
pred_mesh_vertices = np.asarray(pred_mesh.vertices)
pred_mesh_faces = np.asarray(pred_mesh.triangles)

opt_mesh = o3d.geometry.TriangleMesh()


source_offset = optimize_source_to_target(pred_mesh_vertices, npy_shell, args.optimizer_iterations, pred_mesh_faces, lambdas=args.optimizer_params)

opt_mesh.triangles = pred_mesh.triangles
opt_mesh.vertices = o3d.utility.Vector3dVector(pred_mesh_vertices + source_offset)
opt_mesh.compute_vertex_normals()

if args.debug:
    pred_mesh.paint_uniform_color( np.array([0.5, 0.0, 0.5]) )
    pred_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([opt_mesh])


pred_mesh = opt_mesh
    
if args.debug:
    fig_o, axes_o = plt.subplots(8,8,sharex=True, sharey=True)
    vmin = volume.min()
    vmax = volume.max()
    
    for i in range(8):
        for j in range(8):
            ij = (8*i + j)
            vs = volume[:,:,ij]
            axes_o[i,j].imshow( vs, cmap="Blues", vmin=vmin, vmax=vmax )
            axes_o[i,j].set_xticks([])
            axes_o[i,j].set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0) 
    filters.try_all_threshold( vs )
    
    event_images = saved_volume['event_images']
    
    pred_center = vol.get_object_center(args.height_offset, manual_center=args.manual_center).long()
    plt.figure()
    plt.imshow( volume[:,:,pred_center[2]] )
    plt.scatter( pred_center[1], pred_center[0] )
    plt.title("Volume center label " + str(pred_center))

    print(" Primary index ", selected_idx)

    cc_mask = (cc_vol == selected_idx)
    print( cc_mask.sum() )
    multi_slice_viewer(cc_mask)
    
    plt.show()

if args.mesh:
    gt_mesh = o3d.io.read_triangle_mesh(args.mesh)
    gt_mesh_vertices = np.asarray(gt_mesh.vertices) * 0.001
    gt_mesh_vertices = gt_mesh_vertices
    gt_mesh.vertices = o3d.utility.Vector3dVector(gt_mesh_vertices)

    gt_mesh.compute_vertex_normals()
    gt_mesh.paint_uniform_color([1, 0.706, 0])

    if args.debug:
        o3d.visualization.draw_geometries([pred_mesh, gt_mesh])

if args.mesh:
    gt = gt_mesh.sample_points_uniformly(number_of_points=1000)
    carved = pred_mesh.sample_points_uniformly(number_of_points=1000)

    if args.debug:
        o3d.visualization.draw_geometries([pred_mesh, gt_mesh])

    if args.mesh_config is None:
        results = doICP( np.array(gt.points).T, np.array(carved.points).T, 5000, False, False )

        Cx = results[0][-1]
        Cy = results[1][-1]
        Rx = results[2][-1]
        gt_mesh.translate(-Cx)
        pred_mesh.translate(-Cy)

        Tx = np.eye(4)
        Tx[:3,:3] = Rx
        gt_mesh.transform(Tx)
    else:
        with open(args.mesh_config, 'r') as f:
            mesh_config_raw = yaml.load(f)
        
        mesh_pose = np.eye(4)
        mesh_pose[0,3] = mesh_config_raw['pose']['position']['x']
        mesh_pose[1,3] = mesh_config_raw['pose']['position']['y']
        mesh_pose[2,3] = mesh_config_raw['pose']['position']['z']
        
        from scipy.spatial.transform import Rotation as R
        q_x = mesh_config_raw['pose']['orientation']['x']
        q_y = mesh_config_raw['pose']['orientation']['y']
        q_z = mesh_config_raw['pose']['orientation']['z']
        q_w = mesh_config_raw['pose']['orientation']['w']
        import tf_conversions
        mesh_pose[:3,:3] = tf_conversions.transformations.quaternion_matrix([q_x, q_y, q_z, q_w])[:3,:3]
        gt_mesh.transform(mesh_pose)

    if args.debug:
        print("Gt mesh size ", gt_mesh.get_max_bound() - gt_mesh.get_min_bound())
        print("Computed mesh size ", pred_mesh.get_max_bound() - pred_mesh.get_min_bound())
        o3d.visualization.draw_geometries([pred_mesh, gt_mesh])
        o3d.visualization.draw_geometries([pred_mesh])
        o3d.visualization.draw_geometries([gt_mesh])

    gt_pts = gt_mesh.sample_points_uniformly(number_of_points=100000)
    gt_pts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5))
    gt_pts.normalize_normals()
    gt_xyz = np.array(gt_pts.points)
    gt_normal = np.array(gt_pts.normals)

    pred_pts = pred_mesh.sample_points_uniformly(number_of_points=100000)
    pred_pts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5))
    pred_pts.normalize_normals()
    pred_xyz = np.array(pred_pts.points)
    pred_normal = np.array(pred_pts.normals)

    #

    min_score = np.inf
    normal_score = np.inf
    for offset in np.linspace(-0.05, 0.05, 20):
        output = compute_chamfer(gt_xyz, pred_xyz + np.array([[0, 0, offset]]), gt_normal, pred_normal)
        cf_score = output['chamfer']
        cos = output['cos']
        if cf_score < min_score:
            min_score = cf_score
            normal_score = cos

    print("Chamfer score is: {}, and cosine sim is {}".format(min_score, normal_score))

    if args.figure_config is not None:
        image_folder = args.file[:-4] + args.exp
        import os, cv2
        os.makedirs(image_folder, exist_ok=True)

        print(image_folder)

        with open(args.figure_config, 'r') as f:
            import ruamel.yaml as yaml
            fig_conf = yaml.load(f)

        # align with voxel grid
        center = gt_mesh.get_center()
        center[2] = pred_mesh.get_min_bound()[2]
        # pred_mesh.translate(-center)
        # gt_mesh.translate(-center)

        gt_mesh_img, vis = get_figure_image(gt_mesh, fig_conf)
        pred_mesh_img, vis = get_figure_image(pred_mesh, fig_conf, vis)
        comb_mesh_img, vis = get_figure_image([pred_mesh, gt_mesh], fig_conf, vis)
        vol_img, vis = get_figure_image(o3d_volume, fig_conf, vis)

        cv2.imwrite( os.path.join(image_folder, "gt_mesh.png"), gt_mesh_img )
        cv2.imwrite( os.path.join(image_folder, "pred_mesh.png"), pred_mesh_img )
        cv2.imwrite( os.path.join(image_folder, "comb_mesh.png"), comb_mesh_img )
        cv2.imwrite( os.path.join(image_folder, "pred_grid.png"), vol_img )
        o3d.io.write_triangle_mesh(os.path.join(image_folder, "computed_mesh.ply"), pred_mesh)
        with open(os.path.join(image_folder, "results.txt"), 'w') as f:
            f.write("chamfer {}".format(min_score))
            f.write(" normal {}".format(normal_score))

        with open(os.path.join("results", "moec_objects" + args.exp + ".txt"), 'a') as f:
            f.write("{},{},{}\n".format(os.path.basename(args.mesh), min_score, normal_score))
else:
    o3d.visualization.draw_geometries([pred_mesh])
