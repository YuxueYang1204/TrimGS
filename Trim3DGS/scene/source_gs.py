import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.image_utils import generate_grid
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from scene.gaussian_model import GaussianModel

class SourceGaussian(GaussianModel):

	def __init__(self, sh_degree, max_scale, projection_matrix, camera2world, device):
		super().__init__(sh_degree=sh_degree)
		self.active_sh_degree = sh_degree
		self.max_scale = max_scale
		self._depth_map = torch.empty(0)
		self._depth_offset = torch.empty(0)
		self.source_width = 0
		self.source_height = 0
		self.projection_matrix = projection_matrix
		self.camera2world = camera2world
		self.device = device

		self.scaling_activation = torch.sigmoid

	def capture(self):
		return (
			self.active_sh_degree,
			self._depth_map,
			self._depth_offset,
			self._features_dc,
			self._features_rest,
			self._scaling,
			self._opacity,
			self.optimizer.state_dict(),
		)

	def restore(self, model_args, training_args):
		(self.active_sh_degree,
		self._depth_map,
		self._depth_offset,
		self._features_dc,
		self._features_rest,
		self._scaling,
		self._opacity,
		opt_dict) = model_args
		self.training_setup(training_args)
		self.optimizer.load_state_dict(opt_dict)

	@property
	def get_scaling(self):
		return self.max_scale * self.scaling_activation(self._scaling).repeat(1, 3)

	@property
	def get_depth(self):
		return self._depth_map + self._depth_offset

	@property
	def get_xyz(self):
		grid = generate_grid(0.5 / self.source_width, 1 - 0.5 / self.source_width, self.source_width,
					   0.5 / self.source_height, 1 - 0.5 / self.source_height, self.source_height, device=self.device)
		depth = self.get_depth.view(-1, 1)
		# pixel to NDC
		pos = 2 * grid - 1
		# NDC to camera space
		pos[:, 0:1] = (pos[:, 0:1] - self.projection_matrix[2, 0]) * depth / self.projection_matrix[0, 0]
		pos[:, 1:2] = (pos[:, 1:2] - self.projection_matrix[2, 1]) * depth / self.projection_matrix[1, 1]
		pos_camera = torch.cat([pos, depth, torch.ones_like(depth)], dim=-1)
		# camera space to world space
		points = pos_camera @ self.camera2world
		return points[:, :3]

	def create_from_RGBD(self, image, depth_map):
		self.source_width = image.shape[2]
		self.source_height = image.shape[1]
		self._depth_map = depth_map
		self._depth_offset = nn.Parameter(torch.zeros_like(depth_map).requires_grad_(True))

		colors = image.flatten(1).T
		fused_point_cloud = self.get_xyz
		fused_color = RGB2SH(colors)
		features = torch.zeros((fused_color.shape[0], 3, (self.active_sh_degree + 1) ** 2), dtype=torch.float, device=self.device)
		features[:, :3, 0 ] = fused_color
		features[:, 3:, 1:] = 0.0

		print("Number of points at initialisation : ", fused_point_cloud.shape[0])

		scales = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device)
		rot = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float, device=self.device)
		rot[:, 0] = 1
		self._rotation = rot

		opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

		self._features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
		self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
		self._scaling = nn.Parameter(scales.requires_grad_(True))
		self._opacity = nn.Parameter(opacities.requires_grad_(True))

	def training_setup(self, training_args):
		l = [
			{'params': [self._depth_offset], 'lr': training_args.depth_lr_init, "name": "depth_offset"},
			{'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
			{'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
			{'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
		]

		self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
		self.depth_scheduler_args = get_expon_lr_func(lr_init=training_args.depth_lr_init,
													  lr_final=training_args.depth_lr_final,
													  lr_delay_mult=training_args.depth_lr_delay_mult,
													  max_steps=training_args.depth_lr_max_steps)

	def update_learning_rate(self, iteration):
		''' Learning rate scheduling per step '''
		for param_group in self.optimizer.param_groups:
			if param_group["name"] == "depth_offset":
				lr = self.depth_scheduler_args(iteration)
				param_group['lr'] = lr
				return lr

	def construct_list_of_attributes(self):
		l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
		# All channels except the 3 DC
		for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
			l.append('f_dc_{}'.format(i))
		for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
			l.append('f_rest_{}'.format(i))
		l.append('opacity')
		for i in range(self._scaling.shape[1] * 3):
			l.append('scale_{}'.format(i))
		for i in range(self._rotation.shape[1]):
			l.append('rot_{}'.format(i))
		return l

	def save_ply(self, path):
		mkdir_p(os.path.dirname(path))

		xyz = self.get_xyz.detach().cpu().numpy()
		normals = np.zeros_like(xyz)
		f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
		f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
		opacities = self._opacity.detach().cpu().numpy()
		scale = self._scaling.detach().cpu().numpy().repeat(3, axis=1)
		rotation = self._rotation.cpu().numpy()

		dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

		elements = np.empty(xyz.shape[0], dtype=dtype_full)
		attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
		elements[:] = list(map(tuple, attributes))
		el = PlyElement.describe(elements, 'vertex')
		PlyData([el]).write(path)