# training script for TnT dataset
# adapted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_tnt.py

import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import torch

scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck', 'Meetingroom', 'Courthouse']

lambda_dist = {
    'Barn': 100, 'Caterpillar': 100, 'Ignatius': 100, 'Truck': 100,
    'Meetingroom': 10, 'Courthouse': 10
}

voxel_size = {
    'Barn': 0.004, 'Caterpillar': 0.004, 'Ignatius': 0.004, 'Truck': 0.004,
    'Meetingroom': 0.006, 'Courthouse': 0.006
}

sdf_trunc = {
    'Barn': 0.016, 'Caterpillar': 0.016, 'Ignatius': 0.016, 'Truck': 0.016,
    'Meetingroom': 0.024, 'Courthouse': 0.024
}

depth_trunc = {
    'Barn': 3.0, 'Caterpillar': 3.0, 'Ignatius': 3.0, 'Truck': 3.0,
    'Meetingroom': 4.5, 'Courthouse': 4.5
}

excluded_gpus = set([])

output_dir = "output/TNT_2DGS"
tune_output_dir = f"output/TNT_Trim2DGS"
iteration = 7000
split = "scale"
extra_cmd = "--position_lr_init 0.0000016 --contribution_prune_interval 300 --opacity_reset_interval 99999 --depth_grad_thresh 0.0"
jobs = scenes

def train_scene(gpu, scene):
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/TNT_GOF/TrainingSet/{scene} -m {output_dir}/{scene} --eval --test_iterations -1 --quiet --depth_ratio 1.0 -r 2 --lambda_dist {lambda_dist[scene]}",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s data/TNT_GOF/TrainingSet/{scene} -m {tune_output_dir}/{scene} --eval --pretrained_ply {output_dir}/{scene}/point_cloud/iteration_30000/point_cloud.ply --test_iterations -1 --quiet --depth_ratio 1.0 -r 2 --lambda_dist {lambda_dist[scene]} --split {split} {extra_cmd}",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration {iteration} -m {tune_output_dir}/{scene} --quiet --depth_ratio 1.0 --num_cluster 1 --voxel_size {voxel_size[scene]} --sdf_trunc {sdf_trunc[scene]} --depth_trunc {depth_trunc[scene]}",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {tune_output_dir}/{scene}",
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True


def worker(gpu, scene):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene)
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
            future = executor.submit(worker, gpu, job)  # Unpacking job as arguments to worker
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

