import cv2
import numpy as np
import pickle
import torch
import tinycudann as tcnn
import os
from scene.cameras import Camera
from scene.source_gs import SourceGaussian
from utils.image_utils import compute_gradient

class View(Camera):
	def __init__(self, root_path, filename, field_cfg, is_source=False, sh_degree=3, max_scale=0.015, device="cuda"):
		with open(os.path.join(root_path, "viewdata", filename), 'rb') as f:
			view_data = pickle.load(f)
		super().__init__(colmap_id=view_data["colmap_id"],
						 R=view_data["R"],
						 T=view_data["T"],
						 FoVx=view_data["FoVx"],
						 FoVy=view_data["FoVy"],
						 image=view_data["original_image"],
						 gt_alpha_mask=None,
						 image_name=view_data["image_name"],
						 uid=view_data["uid"],
						 data_device=device)
		self.is_source = is_source
		self.device = device
		self.field = tcnn.NetworkWithInputEncoding(n_input_dims=field_cfg["input_dims"], n_output_dims=field_cfg["output_dims"], encoding_config=field_cfg["encoding"], network_config=field_cfg["network"]).to(device)
		ckpt_filename = filename.replace(".pkl", "_field.pth")
		ckpt_path = os.path.join(root_path, "field", ckpt_filename)
		if os.path.exists(ckpt_path):
			self.field.load_state_dict(torch.load(ckpt_path, map_location=device))
			for param in self.field.parameters():
				param.requires_grad = False
		if is_source:
			self.camera2world = self.world_view_transform.inverse()
			self.depth = view_data["depth"].to(device)
			gradient_x, gradient_y = compute_gradient(self.depth[None, None])
			self.depth_grad = torch.norm(torch.cat((gradient_x, gradient_y), dim=1), dim=1)[0]
			self.init_gs(sh_degree, max_scale)

	def init_gs(self, sh_degree, max_scale):
		self.gaussians = SourceGaussian(sh_degree=sh_degree,
										max_scale=max_scale,
								 		projection_matrix=self.projection_matrix,
										camera2world=self.camera2world,
										device=self.device)
		self.gaussians.create_from_RGBD(self.original_image, self.depth)

	def save(self, model_path, iteration):
		point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
		self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

	def project_to_2D(self, pos3D):
		if pos3D.shape[-1] == 3:
			pos3D = torch.cat((pos3D, torch.ones_like(pos3D[..., :1])), dim=-1)
		# world space to NDC
		pos_ndc = pos3D @ self.full_proj_transform
		pos_ndc = pos_ndc / pos_ndc[..., 3:4]
		# NDC to pixel
		pos_pixel = (pos_ndc[..., :2] + 1.0) * 0.5
		return pos_pixel

	def render_depth(self, max_depth=7.0):
		depth = self.gaussians.get_depth
		depth = torch.clamp_max(depth, max_depth)
		depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
		depth = depth.cpu().numpy().astype(np.uint8)
		depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
		return depth