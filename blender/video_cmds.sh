python generate_video.py --debug_id=0
python generate_video.py --debug_id=1

python generate_video.py --debug_id=1 --mesh_type=mesh_dtu --is_texture
python generate_video.py --debug_id=1 --mesh_type=mesh_dtu
python generate_video.py
python generate_video.py --is_texture --mesh_type=mesh_mip

# python generate_video.py --debug_id=2 --mesh_type=mesh_dtu --debug_name=1-25times_fov


python generate_video.py --fps=60 --mesh_type=mesh_dtu --load_dir=/workspace/data/blender/render_res/fov1-25 --mesh_type=mesh_dtu --is_texture
python generate_video.py --fps=60 --mesh_type=mesh_dtu --load_dir=/workspace/data/blender/render_res/fov1-5 --mesh_type=mesh_dtu --is_texture
