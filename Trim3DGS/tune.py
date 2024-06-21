#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, FlattenGaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, blur, generate_grid, compute_gradient
from utils.general_utils import build_rotation
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, FinetuneParams
import logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from pdb import set_trace
from torchvision.utils import save_image

import time

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

def prune_low_contribution_gaussians(gaussians, cameras, pipe, bg, K=5, prune_ratio=0.1):
    top_list = [None, ] * K
    for i, cam in enumerate(cameras):
        trans = render(cam, gaussians, pipe, bg, record_transmittance=True)
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()


def normal_regularization(viewpoint_cam, gaussians, pipe, bg, visibility_filter, depth_grad_thresh=-1.0, close_thresh=1.0, dilation=2, depth_grad_mask_dilation=0):
    pos3D = gaussians.get_xyz
    pos3D = torch.cat((pos3D, torch.ones_like(pos3D[:, :1])), dim=1) @ viewpoint_cam.world_view_transform
    gs_camera_z = pos3D[:, 2:3]
    depth_map = render(viewpoint_cam, gaussians, pipe, bg, override_color=gs_camera_z.repeat(1, 3))["render"][0]
    if depth_grad_thresh > 0:
        depth_grad_x, depth_grad_y = compute_gradient(depth_map[None, None])
        depth_grad_mag = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2).squeeze()
        depth_grad_weight = (depth_grad_mag < depth_grad_thresh).float()
        if depth_grad_mask_dilation > 0:
            mask_di = depth_grad_mask_dilation
            depth_grad_weight = -1 * F.max_pool2d(-1 * depth_grad_weight[None, None, ...], mask_di * 2 + 1, stride=1, padding=mask_di).squeeze()

    grid = generate_grid(
        0.5 / depth_map.shape[-1], 1 - 0.5 / depth_map.shape[-1], depth_map.shape[-1],
        0.5 / depth_map.shape[-2], 1 - 0.5 / depth_map.shape[-2], depth_map.shape[-2], depth_map.device
    )
    depth = depth_map.view(-1, 1)
    # pixel to NDC
    pos = 2 * grid - 1
    # NDC to camera space
    pos[:, 0:1] = (pos[:, 0:1] - viewpoint_cam.projection_matrix[2, 0]) * depth / viewpoint_cam.projection_matrix[0, 0]
    pos[:, 1:2] = (pos[:, 1:2] - viewpoint_cam.projection_matrix[2, 1]) * depth / viewpoint_cam.projection_matrix[1, 1]
    pos_world = torch.cat((pos, depth, torch.ones_like(depth)), dim=-1) @ viewpoint_cam.world_view_transform.inverse()
    pos_world = pos_world[:, :3].permute(1, 0).view(1, 3, *depth_map.shape[-2:])
    pad_pos = F.pad(pos_world, (dilation, ) * 4, mode='replicate')
    di = dilation
    di2x = dilation * 2
    vec1 = pad_pos[:, :, di2x:, di2x:] - pad_pos[:, :, :-di2x, :-di2x]
    vec2 = pad_pos[:, :, :-di2x, di2x:] - pad_pos[:, :, di2x:, :-di2x]
    normal1 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
    vec1 = pad_pos[:, :, di:-di, di2x:] - pad_pos[:, :, di:-di, :-di2x]
    vec2 = pad_pos[:, :, :-di2x, di:-di] - pad_pos[:, :, di2x:, di:-di]
    normal2 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
    normal = F.normalize(normal1 + normal2, p=2, dim=0)
    dir_pp = (viewpoint_cam.camera_center.view(1, 3, 1, 1) - pos_world)
    dir_pp_normalized = F.normalize(dir_pp, p=2, dim=1).squeeze()
    normal = normal * torch.sign((normal * dir_pp_normalized).sum(0, keepdim=True))

    # normal_cam = (normal.flatten(1).T @ viewpoint_cam.world_view_transform[:3, :3]).T.view(3, *depth_map.shape[-2:])

    gs_normal = gaussians.get_normal
    dir_pp = (viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1) - gaussians.get_xyz)
    dir_pp_normalized = F.normalize(dir_pp, p=2, dim=1)
    gs_normal = gs_normal * torch.sign((gs_normal * dir_pp_normalized).sum(1, keepdim=True))
    pred_normal = render(viewpoint_cam, gaussians, pipe, bg, override_color=gs_normal)["render"]
    pred_normal = F.normalize(pred_normal, p=2, dim=0)
    # pred_normal_cam = (pred_normal.flatten(1).T @ viewpoint_cam.world_view_transform[:3, :3]).T.view(3, *depth_map.shape[-2:])

    pred_normal_shift = (pred_normal + 1) / 2
    normal_shift = (normal + 1) / 2

    if depth_grad_thresh > 0:
        normal_loss = (torch.abs(pred_normal_shift - normal_shift) * depth_grad_weight[None]).mean()
    else:
        normal_loss = torch.abs(pred_normal_shift - normal_shift).mean()

    valid_indices = torch.nonzero(visibility_filter).squeeze()
    pos2D = pos3D @ viewpoint_cam.projection_matrix
    ndc_coords = pos2D[:, :2] / pos2D[:, 3:4]
    gs_depthmap_z = F.grid_sample(depth_map[None, None], ndc_coords[valid_indices][None, None], align_corners=True).squeeze()
    close_mask = (gs_depthmap_z - gs_camera_z[valid_indices, 0]).abs() < close_thresh
    valid_indices = valid_indices[close_mask]

    closest_indices = gaussians.knn_idx[valid_indices]
    valid_normal = gs_normal[valid_indices]
    valid_shift_normal = (valid_normal + 1) / 2
    closest_normal = gs_normal[closest_indices]
    closest_normal = closest_normal * torch.sign((closest_normal * valid_normal[:, None]).sum(dim=-1, keepdim=True)).detach()
    closest_mean_shift_normal = (F.normalize(closest_normal.mean(1), p=2, dim=-1) + 1) / 2
    smooth_loss = torch.abs(valid_shift_normal - closest_mean_shift_normal).mean()
    return normal_loss + smooth_loss, normal

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, pretrained_ply, split):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    model_params = "\n".join([f'{k}: {v}' for k, v in vars(dataset).items()])
    logging.info(f"Model params:\n{model_params}")
    fine_tune_params = "\n".join([f'{k}: {v}' for k, v in vars(opt).items()])
    logging.info(f"Finetune Params:\n{fine_tune_params}")
    pipe_params = "\n".join([f'{k}: {v}' for k, v in vars(pipe).items()])
    logging.info(f"Pipeline params:\n{pipe_params}")
    gaussians = FlattenGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, pretrained_ply_path=pretrained_ply)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    all_cameras = scene.getTrainCameras().copy()
    gaussians.reset_neighbor(opt.knn_to_track)
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # # scale regularizatoin
        # scale_reg_loss = scale_regularization(gaussians, iteration, opt)
        # loss = loss + scale_reg_loss

        if iteration > opt.normal_regularity_from_iter:
            normal_reg_loss, render_pkg["normal"] = normal_regularization(
                viewpoint_cam, gaussians, pipe, torch.zeros_like(bg), visibility_filter,
                depth_grad_thresh=opt.depth_grad_thresh,
                close_thresh=opt.normal_close_thresh,
                dilation=opt.normal_dilation,
                depth_grad_mask_dilation=opt.depth_grad_mask_dilation)
            if torch.isnan(normal_reg_loss).any():
                print('Got NaN in normal loss, skip')
                normal_reg_loss = 0
            loss += opt.normal_regularity_param * normal_reg_loss


        loss.backward()

        iter_end.record()


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % opt.neighbor_reset_interval == 0:
                gaussians.reset_neighbor(opt.knn_to_track)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if split == "ordinary":
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, max_screen_size=20)

                    elif split == "scale":
                        scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                        gaussians.densify_and_scale_split(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 100, opt.densify_scale_factor, scene_mask, N=3, no_grad=True)

                    elif split == "mix":
                        size_threshold = 20
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                        gaussians.densify_and_scale_split(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 100, opt.densify_scale_factor, scene_mask, N=3, no_grad=True)
                        prune_mask = (gaussians.get_opacity < 0.005).squeeze()
                        if size_threshold:
                            big_points_vs = gaussians.max_radii2D > size_threshold
                            big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                        gaussians.prune_points(prune_mask)

                    elif split == "none":
                        pass
                    else:
                        raise ValueError(f"Unknown split type {split}")

                    if iteration > opt.contribution_prune_from_iter and iteration % opt.contribution_prune_interval == 0:
                        prune_low_contribution_gaussians(gaussians, all_cameras, pipe, background, K=5, prune_ratio=opt.contribution_prune_ratio)
                        print(f'Num gs after contribution prune: {len(gaussians.get_xyz)}')
                    gaussians.reset_neighbor(opt.knn_to_track)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    logging.basicConfig(filename=os.path.join(args.model_path, "training.log"), level=logging.INFO)
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = FinetuneParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--pretrained_ply", type=str, default = None)
    parser.add_argument("--split", type=str, default = "mix")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations, args.checkpoint_iterations,
        args.start_checkpoint, args.debug_from, args.pretrained_ply, args.split)

    # All done
    print("\nTraining complete.")

