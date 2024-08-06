# training script for DTU dataset
# adapted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_dtu.py

import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import torch

scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

factors = [2] * len(scenes)

excluded_gpus = set([])

output_dir = "output/DTU_Trim2DGS"
iteration = 30000

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    scan_id = scene[4:]
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train_TrimGS.py -s data/dtu_dataset/DTU/{scene} -m {output_dir}/{scene} --quiet --test_iterations -1 --depth_ratio 1.0 -r {factor} --lambda_dist 1000 --split mix --contribution_prune_interval 100",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -s data/dtu_dataset/DTU/{scene} -m {output_dir}/{scene} --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --iteration {iteration} --voxel_size 0.004 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_dtu/evaluate_single_scene.py --input_mesh {output_dir}/{scene}/train/ours_{iteration}/fuse_post.ply --scan_id {scan_id} --output_dir {output_dir}/{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_dtu_pcd/evaluate_single_scene.py --input_pcd {output_dir}/{scene}/point_cloud/iteration_{iteration}/point_cloud.ply --scan_id {scan_id} --output_dir {output_dir}/{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.

def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(range(torch.cuda.device_count()))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

