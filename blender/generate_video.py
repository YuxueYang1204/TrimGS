import imageio
import os
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description='video generator arg parser')
parser.add_argument('--load_dir', type=str, default="render_res")
parser.add_argument("--is_texture", action="store_true")
parser.add_argument("--mesh_type", type=str, default=None)
parser.add_argument("--fps", type=int, default=60)
args = parser.parse_args()

if not args.is_texture:
    write_root = os.path.join(args.load_dir, 'videos', 'mesh')
    args.load_dir = os.path.join(args.load_dir, 'mesh')
else:
    write_root = os.path.join(args.load_dir, 'videos', 'texture')
    args.load_dir = os.path.join(args.load_dir, 'texture')

# if args.debug_id >= 0:
#     # args.load_dir = os.path.join(args.load_dir, "debug")
#     write_root = os.path.join(write_root, f"debug_{args.debug_id}")
# else:
#     write_root = os.path.join(write_root, 'all')


traj_list = os.listdir(args.load_dir)
for traj in traj_list:
    if args.mesh_type is not None and traj != args.mesh_type:
        continue
    root_dir = os.path.join(args.load_dir, traj)
    if not os.path.isdir(root_dir):
        continue
    mesh_list = os.listdir(root_dir)
    for mesh in mesh_list:
        sub_dir = os.path.join(root_dir, mesh)
        if not os.path.isdir(sub_dir):
            continue
        write_dir = os.path.join(write_root, traj)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        video = imageio.get_writer(os.path.join(write_dir, f'{mesh}.mp4'), fps=args.fps)
        image_list = sorted(os.listdir(sub_dir))
        for i in tqdm(range(len(image_list)), desc=f"Creating video"):
            path = os.path.join(sub_dir, image_list[i])
            image = imageio.imread(path)
            video.append_data(image)
        video.close()
