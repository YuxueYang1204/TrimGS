import torch
import torch.nn.functional as F
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import open3d.core as o3c
import math

# try:
#     import debugpy
#     debugpy.listen(5678)
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except:
#     print("Debugpy not available")


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def tsdf_fusion(model_path, name, iteration, views, gaussians, pipeline, background, voxel_size=0.004, depth_trunc=6.0, sdf_trunc=0.02, num_cluster=50):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    makedirs(render_path, exist_ok=True)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            # pos3D = gaussians.get_xyz
            # pos3D = torch.cat((pos3D, torch.ones_like(pos3D[:, :1])), dim=1) @ view.world_view_transform
            # gs_camera_z = pos3D[:, 2:3]
            render_pkg = render(view, gaussians, pipeline, background)

            depth = render_pkg["median_depth"]
            rgb = render_pkg["render"]

            if view.gt_alpha_mask is not None:
                depth[(view.gt_alpha_mask < 0.5)] = 0
            # depth[depth>depth_trunc] = 0

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            W = view.image_width
            H = view.image_height
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, 0, 1]]).float().cuda().T
            intrins =  (view.projection_matrix @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=view.image_width,
                height=view.image_height,
                cx = intrins[0,2].item(),
                cy = intrins[1,2].item(),
                fx = intrins[0,0].item(),
                fy = intrins[1,1].item()
            )
            extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())
            camera = o3d.camera.PinholeCameraParameters()
            camera.intrinsic = intrinsic
            camera.extrinsic = extrinsic

            volume.integrate(rgbd, intrinsic=camera.intrinsic, extrinsic=camera.extrinsic)
        mesh = volume.extract_triangle_mesh()

        # write mesh
        o3d.io.write_triangle_mesh(f"{render_path}/mesh.ply", mesh)
        print(f"mesh saved at {render_path}/mesh.ply")
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
        o3d.io.write_triangle_mesh(f"{render_path}/mesh_post.ply", mesh_post)
        print(f"mesh post processed saved at {render_path}/mesh_post.ply")

def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, voxel_size : float, depth_trunc : float, sdf_trunc, num_cluster):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        train_cameras = scene.getTrainCameras()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        tsdf_fusion(dataset.model_path, "tsdf", iteration, train_cameras, gaussians, pipeline, background, voxel_size, depth_trunc, sdf_trunc, num_cluster)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=7000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--depth_trunc", default=6.0, type=float)
    parser.add_argument("--sdf_trunc", default=0.05, type=float)
    parser.add_argument("--num_cluster", default=50, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.voxel_size, args.depth_trunc, args.sdf_trunc, args.num_cluster)