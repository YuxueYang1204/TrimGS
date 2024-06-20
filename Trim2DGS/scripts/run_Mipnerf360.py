# training script for MipNeRF360 dataset
# adapted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_mipnerf360.py

import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import torch

scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]

factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

excluded_gpus = set([])

output_dir = "output/MipNeRF360_2DGS"
tune_output_dir = f"output/MipNeRF360_Trim2DGS"
iteration = 7000
split = "scale"

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/MipNeRF360/{scene} -m {output_dir}/{scene} --eval -i images_{factor} --test_iterations -1 --quiet",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --pretrained_ply {output_dir}/{scene}/point_cloud/iteration_30000/point_cloud.ply --test_iterations -1 --quiet --split {split}",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration {iteration} -m {tune_output_dir}/{scene} --quiet --eval --skip_train",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {tune_output_dir}/{scene}",
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

