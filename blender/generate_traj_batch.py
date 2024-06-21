import argparse
import os
from cam_pose_utils.cam_reader import readPklSceneInfo, save_to_txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--load_dir',
        type=str,
        default='data/mesh_data')
    parser.add_argument(
        '--mesh_type',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='data/cam_pose/')
    parser.add_argument(
        '--debug_mode',
        action='store_true',
        default=False)
    args = parser.parse_args()
    
    for mesh_type in os.listdir(args.load_dir):
        if args.mesh_type is not None and mesh_type != args.mesh_type:
            continue
        mesh_list = os.listdir(os.path.join(args.load_dir, mesh_type))
        for mesh in mesh_list:
            cam_infos = readPklSceneInfo(os.path.join(args.load_dir, mesh_type, mesh, "traj"))
            save_to_txt(cam_infos, os.path.join(args.save_dir, mesh_type, mesh))

# python cam_pose_utils/traj_batch.py --mesh_type=mesh_dtu