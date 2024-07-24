# evaluation script for TnT dataset

from concurrent.futures import ProcessPoolExecutor
import subprocess

scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck', 'Meetingroom', 'Courthouse']

excluded_gpus = set([])

output_dir = "output/TNT_Trim2DGS"
iteration = 7000

jobs = scenes

def eval(scene):
    cmds = [
            f"python scripts/eval_tnt/run.py --dataset-dir data/TNT_GOF/ground_truth/{scene} --traj-path data/TNT_GOF/TrainingSet/{scene}/{scene}_COLMAP_SfM.log --ply-path {output_dir}/{scene}/train/ours_{iteration}/fuse_post.ply",
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def main():
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(eval, scene) for scene in jobs]

        for future in futures:
            try:
                result = future.result()
                print(f"Finished job with result: {result}\n")
            except Exception as e:
                print(f"Failed job with exception: {e}\n")

if __name__ == "__main__":
    main()