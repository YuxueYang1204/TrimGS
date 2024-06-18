import os
import math
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import numpy as np
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.cameras import Camera
from pdb import set_trace
import torch
from torchvision.utils import save_image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import inverse_sigmoid, build_rotation

import torch_scatter
import cv2

mm = torch.matmul

def get_inv_scale(s):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)

    L[:,0,0] = 1 / s[:,0]
    L[:,1,1] = 1 / s[:,1]
    L[:,2,2] = 1 / s[:,2]

    return L

def argsort_3d(coors):
    assert coors.size(1) == 3
    assert (coors >= 0).all()
    x, y, z = coors[:, 0], coors[:, 1], coors[:, 2]
    x_dim, y_dim, z_dim = x.max() + 1, y.max() + 1, z.max() + 1
    coors_1d = x * y_dim * z_dim + y * z_dim + z
    inds = torch.argsort(coors_1d)
    return inds

def voxelize(points, voxel_size, points_min):

    device = points.device
    assert points_min.ndim == 1 and points_min.size(0) == 3

    voxel_size = torch.tensor(voxel_size, device=points.device, dtype=points.dtype)
    # points_min = points.min(0)[0] - 1e-5
    coors = torch.div(points - points_min[None, :], voxel_size[None, :], rounding_mode='floor').int()
    assert (coors >= 0).all()

    return coors


def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_x / 2
    z_max = scene_center[2] + span_x / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')
    
    return valid_mask, scene_center

def estimate_density(xyz, density, master_idx, voxel_size, lower_boundary):
    coors = voxelize(xyz, [voxel_size,] * 3, lower_boundary)
    sub_coors = torch.cat([coors, master_idx[:, None]], dim=1)
    # voxel_density, voxel_coors = scatter(sampled_density, coors)

    unq_sub_coors, unq_inv_sub, cnt = torch.unique(sub_coors, return_inverse=True, return_counts=True, sorted=True, dim=0)

    sub_density = torch_scatter.scatter(density, unq_inv_sub, dim=0, reduce='mean')
    sub_density = sub_density / cnt.float()
    sub_density = sub_density[unq_inv_sub.long()]
    out_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, sorted=True, dim=0)
    out_density = torch_scatter.scatter(sub_density, unq_inv, dim=0, reduce='sum')

    return out_density, out_coors, unq_sub_coors

def get_scene_density(xyz, scale, rot, opacity, num_mc, ignore_opacity, scene_center_y=None, resolution=512):

    sample_xyz, sample_density, sample_master_inds = dynamic_mc_sampling(xyz, scale, rot, opacity, num_mc, ignore_opacity)

    temp = torch.cat([sample_xyz, xyz], dim=0)
    lower_boundary = temp.min(0)[0] - 1e-5
    upper_boundary = temp.max(0)[0] + 1e-5

    voxel_size = (upper_boundary[0] - lower_boundary[0]).item() / resolution # x_max - x_min
    # print(f'voxel_size: {voxel_size}')

    voxel_density, voxel_coors, unq_voxel_master_pair_code = estimate_density(
        sample_xyz, sample_density, sample_master_inds, voxel_size, lower_boundary=lower_boundary
    )

    gs_coors = voxelize(xyz, [voxel_size,] * 3, lower_boundary)

    if scene_center_y is not None:
        center_y_coor = (scene_center_y - lower_boundary[1].item()) // voxel_size
    else:
        center_y_coor = None
    
    voxel_info = {'voxel_size':voxel_size, 'lower_boundary':lower_boundary}

    return voxel_density, voxel_coors, gs_coors, unq_voxel_master_pair_code, center_y_coor, voxel_info

def sample_gs_for_moving(gs_xyz, gs_coors, scene_density, density_coors, ratio):
    m1, m2 = coors_union_mask(gs_coors, density_coors)
    gs_coors = gs_coors[m1]
    gs_xyz = gs_xyz[m1]
    density_coors = density_coors[m2]
    scene_density = scene_density[m2]

    tile = torch.quantile(scene_density, ratio)
    sampled_coors = density_coors[scene_density < tile]
    sampled_gs_mask, _ = coors_union_mask(gs_coors, sampled_coors)
    sampled_xyz = gs_xyz[sampled_gs_mask]
    sampled_gs_coors = gs_coors[sampled_gs_mask]
    return sampled_xyz, sampled_gs_coors

def dynamic_mc_sampling(xyz, scale, rot_q, opacity, num_mc, ignore_opacity=False):
    assert rot_q.size(1) == 4
    device = xyz.device
    num_mc_per_unit = num_mc / scale.max(1)[0].mean()
    # num_each_dim = (scale * num_mc_per_unit).clamp_min(1)
    # num_each_gs = num_each_dim[:, 0] * num_each_dim[:, 1] * num_each_dim[:, 2]
    num_each_gs = (scale.max(1)[0] * num_mc_per_unit).clamp_min(1.1).ceil().int() # make sure at 2 point each gs to make this cumsum based algo work
    num_all_samples = num_each_gs.sum().item()
    cumsum = torch.cumsum(num_each_gs, dim=0)
    group_beg_idx = cumsum - num_each_gs
    group_end_idx = cumsum - 1
    mapping = torch.zeros(num_all_samples, device=device, dtype=torch.long)
    arange = torch.arange(xyz.size(0), dtype=torch.long, device=device)
    mapping[group_beg_idx] = arange.clone()
    mapping[group_end_idx] = -arange.clone()
    # mapping_before_cum = mapping.clone()
    mapping = torch.cumsum(mapping, dim=0)
    assert (mapping[group_end_idx] == 0).all()
    mapping[group_end_idx] = arange

    scale = scale.clamp_min(1e-6) # for stability
    stds = scale[mapping] #[N_all_sample, 3]
    means = torch.zeros((num_all_samples, 3), device=xyz.device)
    samples = torch.normal(mean=means, std=stds) #[N_all_sample, 3]
    rot_mat = build_rotation(rot_q)
    rot_mat_expand = rot_mat[mapping]
    rotated_local_shift = torch.matmul(rot_mat_expand, samples.unsqueeze(-1)).squeeze(-1).detach() # It feels like should be detached
    sample_xyz = xyz[mapping] + rotated_local_shift

    # Method 1: calculate in canonical pose, gradient only pass to scale and opacity
    density = torch.exp(-0.5 * (samples / stds * samples / stds).sum(-1)) 
    if not ignore_opacity:
        density = density * opacity[mapping]
    master_idx = mapping
    return sample_xyz, density, master_idx

def coors_union_mask(coors1, coors2):

    _, unq_inv1, cnt1 = torch.unique(coors1, dim=0, return_inverse=True, return_counts=True)
    old_count_per_ele_1 = cnt1[unq_inv1]

    _, unq_inv2, cnt2 = torch.unique(coors2, dim=0, return_inverse=True, return_counts=True)
    old_count_per_ele_2 = cnt2[unq_inv2]

    _, unq_inv_new, cnt_new = torch.unique(torch.cat([coors1, coors2], 0), dim=0, return_inverse=True, return_counts=True)
    new_count_per_ele = cnt_new[unq_inv_new]

    mask1 = new_count_per_ele[:coors1.shape[0]] > old_count_per_ele_1
    mask2 = new_count_per_ele[coors1.shape[0]:] > old_count_per_ele_2
    return mask1, mask2

def calculate_xyz_density(sample_xyz, xyz, scale, rot, opacity):
    # scale_inv = get_inv_scale(scale)

    shift = xyz - sample_xyz

    rot_mat = build_rotation(rot) #[N, 3, 3]
    cano_shift = torch.bmm(rot_mat.transpose(1, 2), shift[:, :, None]).squeeze(-1)

    density = torch.exp(-0.5 * (cano_shift / scale * cano_shift / scale).sum(1)) * opacity
    return density
    # density = torch.exp(-0.5 * 
    #     (
    #         mm(mm(shift[:, None, :], rot_mat), scale_inv[:, None, :, :]) @
    #         mm((scale_inv @ rot_mat.transpose(1, 2))[:, None, :, :], shift[:, :, :, None])
    #     ).squeeze()
    # ) * opacity[:, None]
def create_density_map(density, coors, resolution):
    assert coors.max().item() < resolution
    canvas = torch.zeros(resolution * resolution, device=density.device, dtype=density.dtype)
    coors_1d = coors[:, 0] * resolution + coors[:, 1]
    density = density / density.max()
    canvas[coors_1d.long()] = density
    canvas = canvas.reshape(resolution, resolution)
    return canvas

def vis_density_map(voxel_density, voxel_coors, center_y_coor, vis_path, resolution=512):
    max_density = torch.quantile(voxel_density, 0.95) # for better visualization
    voxel_density = voxel_density.clamp(max=max_density)

    voxel_density = voxel_density / max_density
    assert (voxel_coors < resolution).all()

    layer_mask = voxel_coors[:, 1] == center_y_coor

    layer_density = voxel_density[layer_mask]
    layer_coors = voxel_coors[layer_mask][:, [0, 2]]

    density_map = create_density_map(layer_density, layer_coors, resolution)
    # save_image(density_map, save_file_name)

    density_map = density_map * 255
    density_map = density_map.cpu().numpy().astype(np.uint8)
    density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_INFERNO)
    cv2.imwrite(vis_path, density_map)


def density_moving_loss(gs, cams, samplign_ratio = 0.1, vis_path=None):
    xyz = gs.get_xyz
    scale = gs.get_scaling
    rot = gs.get_rotation
    opacity = gs.get_opacity.squeeze(-1)
    device = xyz.device

    scene_mask, scene_center = culling(xyz, cams)

    xyz = xyz[scene_mask]
    scale = scale[scene_mask]
    rot = rot[scene_mask]
    opacity = opacity[scene_mask]

    with torch.no_grad():
        scene_density, density_coors, gs_coors, unq_voxel_master_pair_code, center_y_coor, voxel_info = \
            get_scene_density(xyz, scale, rot, opacity, num_mc=10, ignore_opacity=True, scene_center_y=scene_center[1].item())
    # coors to xyz: (density_coors + 0.5) * voxel_info['voxel_size'] + voxel_info['lower_boundary']
    
    if vis_path is not None:
        vis_density_map(scene_density, density_coors, center_y_coor, vis_path) # only show sliced plane at y == center_y_coor

    sampled_xyz, sampled_gs_coors = sample_gs_for_moving(xyz, gs_coors, scene_density, density_coors, samplign_ratio)

    m1, m2 = coors_union_mask(unq_voxel_master_pair_code[:, :3], sampled_gs_coors)

    unq_voxel_master_pair_code = unq_voxel_master_pair_code[m1]
    sampled_gs_coors = sampled_gs_coors[m2]
    sampled_xyz = sampled_xyz[m2]

    voxel_sort_inds = argsort_3d(unq_voxel_master_pair_code[:, :3])
    sorted_pair_code = unq_voxel_master_pair_code[voxel_sort_inds]

    unq_1, unq_inv_voxel, num_master_each_voxel = torch.unique(sorted_pair_code[:, :3], dim=0, return_inverse=True, return_counts=True)

    # sort sampled_gs_coors
    gs_sort_inds = argsort_3d(sampled_gs_coors)
    sampled_gs_coors = sampled_gs_coors[gs_sort_inds]
    sampled_xyz = sampled_xyz[gs_sort_inds]

    assert (sorted_pair_code[0, :3] == sampled_gs_coors[0]).all()

    sampled_gs_coors, unq_inv_gs_coors, grad_weight = torch.unique(sampled_gs_coors, dim=0, return_inverse=True, return_counts=True)
    assert (unq_1 == sampled_gs_coors).all()

    # if multiple sampled gs fall in the same voxel, we use the centroid to supervise density
    # greatly simplify the implementation
    # so we need counts as grad_weight to recover grad scale, which is decreased my mean reduction
    sampled_xyz = torch_scatter.scatter(sampled_xyz, unq_inv_gs_coors, dim=0, reduce='mean') 

    expand_gs_inds = torch.arange(sampled_gs_coors.size(0), device=device)[unq_inv_voxel] # e.g., [0, 0, 1, 1, 2, 3], indexes the sampled_xyz
    assert (expand_gs_inds.sort()[0] == expand_gs_inds).all()
    master_idx_for_expand_gs = sorted_pair_code[:, -1]

    assert expand_gs_inds.size(0) == master_idx_for_expand_gs.size(0)

    expand_sampled_xyz = sampled_xyz[expand_gs_inds]
    grad_weight = grad_weight[expand_gs_inds] # 

    # detach master grad, only push sampled_xyz to denser region
    expand_master_xyz = xyz[master_idx_for_expand_gs].detach()
    expand_master_scale = scale[master_idx_for_expand_gs].detach()
    expand_master_rot = rot[master_idx_for_expand_gs].detach()
    expand_master_opacity = opacity[master_idx_for_expand_gs].detach()


    xyz_density = calculate_xyz_density(
        expand_sampled_xyz,
        expand_master_xyz,
        expand_master_scale,
        expand_master_rot,
        expand_master_opacity
    )
    # set_trace()

    return ((1 - xyz_density) * grad_weight.float()).mean() # minimize density

    





