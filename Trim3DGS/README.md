# Instruction for Trim3DGS

## Installation

```bash
# if you forget to clone the submodules, type the following command
# git submodule update --init --recursive
cd Trim3DGS
conda env create -f environment.yml
conda activate trim3dgs
```

## Dataset Preparation

### DTU

For geometry reconstruction on DTU dataset, please download the [preprocessed data](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) (only `dtu.tar.gz` is needed). You also need to download the ground truth DTU point cloud [SampleSet.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) from the [DTU official website](https://roboimagedata.compute.dtu.dk/?page_id=36).

```bash
cd data/
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
cd data/
mkdir MipNeRF360
# prepare the MipNeRF360 dataset
unzip 360_v2.zip -d MipNeRF360
unzip 360_extra_scenes.zip -d MipNeRF360
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
└── MipNeRF360
    ├── bicycle
    └── ...
```

## Training and Evaluation

```bash
# train DTU dataset, the extracted mesh can be found in output/DTU_Trim3DGS/scan{scene}/tsdf/ours_{iteration}/mesh_post.ply
python scripts/run_dtu.py
# print the evaluation results
python print_results.py -o output/DTU_Trim3DGS --dataset dtu

# MiNeRF360
python scripts/run_Mipnerf360.py
# print the evaluation results
python print_results.py -o output/MipNeRF360_Trim3DGS --dataset mipnerf360
```

