# Instruction for Rendering using bpy

## Installation

```bash
cd blender
conda create -n blender python=3.11 -y 
conda activate blender
pip install bpy==4.1
pip install imageio
pip install tqdm
pip install imageio[ffmpeg]
```

## Trajectory Preparation

To render a video for a targeted mesh `mesh.ply`, please pack the intrinsics and extrinsics of each camera on the desired trajectory into dictionaries with following 4 keys:
    "image_height",
    "image_width",
    "world_view_transform" and
    "FoVx",
and dump them into pickle files naming `XXXXX.pkl`. Then organize the data, making the `blender/data` folder look as follows:
```
```bash
data/mesh_data
        ├── mesh_dtu
        │   ├── scan24
        |   |   ├── mesh.ply
        |   |   └── traj
        |   |       ├── 00000.pkl
        |   |       └── ...
        |   ├── ...
        │   └── scan122
        └── mesh_mip
            ├── bicycle
            ├── ...
            └── ...
```

After that, simply run
```bash
python generate_traj_batch.py
```
to extract the camera parameters. Now the data folder should look like this:

```bash
data
├── mesh_data
|   ├── mesh_dtu
|   └── mesh_mip
└── cam_pose
    ├── mesh_dtu
    └── mesh_mip
```

## Rendering all scenes

### Rendering

With data prepared, run
```bash
python mesh.py --mesh_type mesh_dtu
python mesh.py --mesh_type mesh_mip
```
to render images of all (**vanilla**) meshes on their corresponding trajectories, with results saved in the folder `render_res`. Images of **textured** meshes can be got via 
```bash
python mesh.py --mesh_type mesh_dtu --is_texture
python mesh.py --mesh_type mesh_mip --is_texture
```

### Video Generation
With images rendered, run
```bash
python generate_video.py
```
to concatenate all of them into desired videos.

## Useful Command Line Arguments

### Rendering some scenes

If you want to render only **some** scenes, you can specify them by adding command line arguments. For example, run 
```bash
python mesh.py --mesh_type mesh_dtu --mesh_list scan24 scan40
```
to generate images for `scan24` and `scan40` of `DTU` only.

### Protection of Conflicts
By default, `mesh.py` refuses to re-generate images if the path where the rendered images should be stored already exists. Set `--write_cover` to cover previous results forcibly. 

To avoid the conflict stated above, set `--save_root_dir` in `mesh.py` to save the rendered images in a specific path. Consequently, change `--load_dir` in `generate_video.py` to generate video from that path. For example, run
```bash
python mesh.py --mesh_type mesh_dtu --is_texture --save_root_dir=render_res_test
python generate_video.py --load_dir=render_res_test
```

### Debug Mode

You can specify command line arguments `--debug_mode` and `--debug_video_step` in `mesh.py` to render some images of a scene only, for quickly debugging. 
