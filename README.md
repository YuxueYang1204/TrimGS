# Trim 3D Gaussian Splatting for Accurate Geometry Representation

![Teaser image](assets/teaser.jpg)


More demonstrations can be found on our [project page](https://trimgs.github.io/) and [arXiv](https://arxiv.org/abs/2406.07499) paper.

## Updates

- [24-08-06] We support training Trim2DGS from scratch. The performances on DTU and MipNeRF360 are all improved compared to the results in the paper.
- [24-07-24] Trim2DGS is updated to support TnT dataset.
- [24-06-25] We release the demo video rendering [scripts](https://github.com/YuxueYang1204/TrimGS/tree/main/blender) with Blender to show the mesh.
- [24-06-20] The code of Trim2DGS is released. **We have made Trim3DGS and Trim2DGS compatible in a shared environment for easier use!** If you have installed `trim3dgs` environment following the older instruction, please remove it and reinstall the new environment `trimgs`.
- [24-06-18] The code of Trim3DGS is released.

## Installation

```bash
git clone git@github.com:YuxueYang1204/TrimGS.git --recursive
# Here we comment out the pip install Trim2DGS/submodules/simple-knn in environment.yml, since it is the same as the one in Trim3DGS
conda env create -f environment.yml
conda activate trimgs
```

**Note:** We modify the differentiable rasterization kernel to support contribution-based trimming, which can be found in [diff-gaussian-rasterization (for Trim3DGS)](https://github.com/Abyssaledge/diff-gaussian-rasterization) and [diff-surfel-rasterization (for Trim2DGS)](https://github.com/YuxueYang1204/diff-surfel-rasterization).


## Dataset Preparation

### DTU

For geometry reconstruction on DTU dataset, please download the [preprocessed data](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) (only `dtu.tar.gz` is needed). You also need to download the ground truth DTU point cloud [SampleSet.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) from the [DTU official website](https://roboimagedata.compute.dtu.dk/?page_id=36).

```bash
mkdir data && cd data
mkdir dtu_dataset
# prepare the dtu dataset
tar -xvf dtu.tar.gz -C dtu_dataset
# prepare the ground truth
unzip SampleSet.zip
# remove the incomplete points and unzip the complete points
rm -r SampleSet/MVS\ Data/Points
unzip Points.zip -d SampleSet/MVS\ Data
mv SampleSet/MVS\ Data dtu_dataset/Official_DTU_Dataset
```

### MipNeRF360

For novel view synthesis on MipNeRF360, please download the `360_v2.zip` and `360_extra_scenes.zip` from [MipNeRF360](https://jonbarron.info/mipnerf360/).

```bash
cd data
mkdir MipNeRF360
unzip 360_v2.zip -d MipNeRF360
unzip 360_extra_scenes.zip -d MipNeRF360
```

### TnT

For geometry reconstruction on TnT dataset, please download the preprocessed [TnT_data](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main). You also need to download the ground truth [Results on training set](https://drive.google.com/file/d/1jAr3IDvhVmmYeDWi0D_JfgiHcl70rzVE/view?usp=sharing&resourcekey=).

```bash
cd data
unzip TNT_GOF.zip
unzip trainingdata.zip -d TNT_GOF/ground_truth
```

Due to the requirement conflict of `open3d`, we need to create a new environment for TnT evaluation:

```bash
conda create -n tnt_eval python=3.8 -y
conda activate tnt_eval
pip install open3d==0.10
pip install trimesh
```

Now the data folder should look like this:

```bash
data
├── dtu_dataset
│   ├── DTU
│   │   ├── scan24
│   │   ├── ...
│   │   └── scan122
│   └── Official_DTU_Dataset
│       ├── Points
│       │   └── stl
│       └── ObsMask
├── MipNeRF360
│   ├── bicycle
│   └── ...
└── TNT_GOF
    ├── ground_truth
    │   ├── Barn
    │   └── ...
    └── TrainingSet
        ├── Barn
        └── ...
```

Then link the data folder to the Trim3DGS and Trim2DGS:

```bash
ln -s data Trim3DGS/data
ln -s data Trim2DGS/data
```

## Training and Evaluation

### Trim3DGS

```bash
cd Trim3DGS
# train DTU dataset
python scripts/run_dtu.py
# print the evaluation results
python print_results.py -o output/DTU_Trim3DGS --dataset dtu
# the extracted mesh of DTU dataset can be found in `output/DTU_Trim3DGS/scan${scene_id}/tsdf/ours_7000/mesh_post.ply`

# MiNeRF360
python scripts/run_Mipnerf360.py
# print the evaluation results
python print_results.py -o output/MipNeRF360_Trim3DGS --dataset mipnerf360
```


### Trim2DGS

```bash
cd Trim2DGS
# DTU dataset
# train 2DGS 30K and finetune Trim2DGS 7K
python scripts/run_dtu.py
# or you can train Trim2DGS from scratch
python Trim2DGS/scripts/train_from_scratch/run_dtu.py
# print the evaluation results
python print_results.py -o output/DTU_Trim2DGS --dataset dtu
# the extracted mesh of DTU dataset can be found in `output/DTU_Trim2DGS/scan${scene_id}/train/ours_7000/fuse_post.ply`

# MiNeRF360
# train 2DGS 30K and finetune Trim2DGS 7K
python scripts/run_Mipnerf360.py
# or you can train Trim2DGS from scratch
python Trim2DGS/scripts/train_from_scratch/run_Mipnerf360.py
# print the evaluation results
python print_results.py -o output/MipNeRF360_Trim2DGS --dataset mipnerf360

# TnT
python scripts/run_TNT.py
# evaluate the results
conda activate tnt_eval
python scripts/eval_TNT.py
python print_results.py -o output/TNT_Trim2DGS --dataset tnt
```

---

We provide the meshes of DTU dataset from [Trim3DGS](https://drive.google.com/file/d/1R4m3cz7Be59qEVrXxyUD75UDthMx8thM/view?usp=sharing) and [Trim2DGS](https://drive.google.com/file/d/17yvj0Dh7msePLybkX_nMIIOcAs64bN54/view?usp=sharing) for evaluation.

Thank [hbb1](https://github.com/hbb1) for his kind suggestion on Trim2DGS in https://github.com/YuxueYang1204/TrimGS/issues/1, resulting in a better performance.

<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

Chamfer distance on DTU dataset (lower is better)

|   | 24   | 37   | 40   | 55   | 63   | 65   | 69   | 83   | 97   | 105  | 106  | 110  | 114  | 118  | 122  | Mean |
|----------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 2DGS (Paper)    | 0.48 | 0.91 | 0.39 | 0.39 | 1.01 | 0.83 | 0.81 | 1.36 | 1.27 | 0.76 | 0.70 | 1.40 | 0.40 | 0.76 | 0.52 | 0.80 |
| 2DGS (Reproduce) | 0.46 | 0.80 | 0.33 | 0.37 | 0.95 | 0.86 | 0.80 | 1.25 | 1.24 | 0.67 | 0.67 | 1.24 | 0.39 | 0.64 | 0.47 | 0.74 |
| Trim2DGS (Paper) | 0.48 | 0.82 | 0.44 | 0.45 | 0.95 | 0.75 | 0.74 | 1.18 | 1.13 | 0.72 | 0.70 | 0.99 | 0.42 | 0.62 | 0.50 | 0.72 |
| Trim2DGS (Reproduce) | 0.45 | 0.72 | 0.33 | 0.40 | 0.97 | 0.72 | 0.73 | 1.21 | 1.14 | 0.61 | 0.67 | 1.01 | 0.41 | 0.60 | 0.44 | 0.69 |
</details>


We provide geometry evaluation results of Trim2DGS on TnT dataset test-set.

<details>
<summary><span style="font-weight: bold;">Table Results</span></summary>

F1 scores on TnT dataset (higher is better)

|   | Barn | Caterpillar | Ignatius | Truck | Meetingroom | Courthouse | Mean |
|----------|------|------|------|------|------|------|------|
| 2DGS | 0.30 | 0.25 | 0.54 | 0.50 | 0.19 | 0.13 | 0.32 |
| Trim2DGS | 0.31 | 0.30 | 0.61 | 0.56 | 0.20 | 0.18 | 0.36 |

</details>

## Todo

- [x] Release the code of Trim3DGS.
- [x] Release the code of Trim2DGS.
- [x] Release scripts for making demo videos.
- [x] Support TnT dataset.
- [ ] Integrate TrimGS into more methods such as [GOF](https://niujinshuchong.github.io/gaussian-opacity-fields/).

## Acknowledgements

We sincerely thank the authors of the great works [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and [2D Gaussian Splatting](https://surfsplatting.github.io/). Our Trim3DGS and Trim2DGS are built upon these foundations.

The running scripts are adapted from [GOF](https://niujinshuchong.github.io/gaussian-opacity-fields/) and the evaluation scripts for DTU and Tanks and Temples dataset are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation) respectively.

We thank all the authors for their great work and repos.

## Citation

Please consider citing our work as follows if it is helpful.

```bibtex
@article{fan2024trim,
  author    = {Fan, Lue and Yang, Yuxue and Li, Minxing and Li, Hongsheng and Zhang, Zhaoxiang},
  title     = {Trim 3D Gaussian Splatting for Accurate Geometry Representation},
  journal   = {arXiv preprint arXiv:2406.07499},
  year      = {2024},
}
```
