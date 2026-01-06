# **MGPC: <u>M</u>ultimodal Network for <u>G</u>eneralizable <u>P</u>oint Cloud <u>C</u>ompletion With Modality Dropout and Progressive Decoding**

This is the official repository of the paper "Monocular Depth Estimation and Segmentation for Transparent Object with Iterative Semantic and Geometric Fusion".

## Note

We have released the test code and data for now. The training code and dataset will be released after the paper is accepted.

## Requirements

We have tested on Ubuntu 20.04/22.04 with NVIDIA GeForce RTX 4090 with Python 3.10 and cuda12.1. The code may work on other systems. 

## Installation

- **Setup a virtual environment**

  ``````bash
  python3.10 -m venv mgpc
  source mgpc/bin/activate
  ``````

- **Install pip dependencies**

  Note that other versions of Pytorch may also work.

  ``````bash
  cd MGPC
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ``````

- **Build extensions for Chamfer Distance and PointNet++**

  ``````bash
  cd extensions/Chamfer3D && python setup.py install
  cd ../pointnet2_ops_lib && python setup.py install
  cd ../..
  ``````

- **Download the datasets**

  The test set of our MGPC_1M dataset can be downloaded from [here](https://drive.google.com/drive/folders/12LGvcBvQkxgiNLPnQNcZFSoJzv2lIhYn?usp=drive_link).

- **Download the model weight**

​	We will provide our pre-trained weights in a few days.

- **Modify the configuration file**

  Modify the arguments in the corresponding script. Specify all paths, batch size, and so on. 

## Evaluation

To evaluate the model on the test set, run:

```bash
python test.py
```

## Inference

We provide some in-the-wild data in `/demo`, and you can also collect the data by yourselves. To test the zero-shot generalization, run:

```bash
python demo.py
```

## Acknowledgement

Our code is partially built upon [PoinTr](https://github.com/yuxumin/PoinTr), [SeedFormer](https://github.com/hrzhou2/seedformer), [TripoSR](https://github.com/VAST-AI-Research/TripoSR), and [PUCRN](https://github.com/hikvision-research/3DVision). We thank them for their nicely open sourced code and their great contributions to the community.

## Citation

