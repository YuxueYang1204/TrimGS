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

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.image_utils import generate_grid
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_other=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if render_other:
        normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
        depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
        makedirs(normal_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        outputs = render(view, gaussians, pipeline, background)
        rendering = outputs["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{view.image_name}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{view.image_name}.png"))

        if not render_other:
            continue
        gt_alpha_mask = torch.zeros_like(rendering[0], dtype=torch.bool) if view.gt_alpha_mask is None else (view.gt_alpha_mask.squeeze() < 0.5)
        assert gt_alpha_mask.ndim == 2

        # pos3D = gaussians.get_xyz
        # pos3D = torch.cat((pos3D, torch.ones_like(pos3D[:, :1])), dim=1) @ view.world_view_transform
        # gs_camera_z = pos3D[:, 2:3]
        # depth_map = render(view, gaussians, pipeline, torch.zeros_like(background), override_color=gs_camera_z.repeat(1, 3))["render"][0:1]
        depth_map = outputs["median_depth"]
        depth_map[..., gt_alpha_mask] = 0
        torchvision.utils.save_image((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()), os.path.join(depth_path, f"{view.image_name}.png"))
        gs_normal = gaussians.get_normal
        dir_pp = (view.camera_center.repeat(gaussians.get_features.shape[0], 1) - gaussians.get_xyz)
        dir_pp_normalized = F.normalize(dir_pp, p=2, dim=1)
        gs_normal = gs_normal * torch.sign((gs_normal * dir_pp_normalized).sum(1, keepdim=True))
        normal_map = render(view, gaussians, pipeline, torch.zeros_like(background), override_color=gs_normal)["render"]
        normal_map = F.normalize(normal_map, p=2, dim=0)
        normal_map[..., gt_alpha_mask] = 0
        torchvision.utils.save_image((normal_map + 1) / 2, os.path.join(normal_path, f"{view.image_name}_world.png"))
        normal_map = (normal_map.flatten(1).T @ view.world_view_transform[:3, :3]).T.view(3, *normal_map.shape[-2:])
        normal_map[..., gt_alpha_mask] = 0
        torchvision.utils.save_image((normal_map + 1) / 2, os.path.join(normal_path, f"{view.image_name}_view.png"))

        grid = generate_grid(
            0.5 / depth_map.shape[-1], 1 - 0.5 / depth_map.shape[-1], depth_map.shape[-1],
            0.5 / depth_map.shape[-2], 1 - 0.5 / depth_map.shape[-2], depth_map.shape[-2], depth_map.device
        )
        depth = depth_map.view(-1, 1)
        # pixel to NDC
        pos = 2 * grid - 1
        # NDC to camera space
        pos[:, 0:1] = (pos[:, 0:1] - view.projection_matrix[2, 0]) * depth / view.projection_matrix[0, 0]
        pos[:, 1:2] = (pos[:, 1:2] - view.projection_matrix[2, 1]) * depth / view.projection_matrix[1, 1]
        pos_world = torch.cat((pos, depth, torch.ones_like(depth)), dim=-1) @ view.world_view_transform.inverse()
        pos_world = pos_world[:, :3].permute(1, 0).view(1, 3, *depth_map.shape[-2:])
        pad_pos = F.pad(pos_world, (2, 2, 2, 2), mode='replicate')
        vec1 = pad_pos[:, :, 4:, 4:] - pad_pos[:, :, :-4, :-4]
        vec2 = pad_pos[:, :, :-4, 4:] - pad_pos[:, :, 4:, :-4]
        normal1 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
        vec1 = pad_pos[:, :, 2:-2, 4:] - pad_pos[:, :, 2:-2, :-4]
        vec2 = pad_pos[:, :, :-4, 2:-2] - pad_pos[:, :, 4:, 2:-2]
        normal2 = F.normalize(torch.cross(vec1, vec2, dim=1), p=2, dim=1)[0]
        normal = F.normalize(normal1 + normal2, p=2, dim=0)
        dir_pp = (view.camera_center.view(1, 3, 1, 1) - pos_world)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        reverse_mask = ((normal * dir_pp_normalized).sum(1) < 0).squeeze(0)
        normal[..., reverse_mask] = -normal[..., reverse_mask]
        normal = (normal.flatten(1).T @ view.world_view_transform[:3, :3]).T.view(3, *depth_map.shape[-2:])
        normal[..., gt_alpha_mask] = 0
        torchvision.utils.save_image((normal + 1) / 2, os.path.join(normal_path, f"{view.image_name}_view_from_depth.png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_other : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_other)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_other)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_other", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_other)