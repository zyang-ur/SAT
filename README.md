# SAT: 2D Semantics Assisted Training for 3D Visual Grounding
[SAT: 2D Semantics Assisted Training for 3D Visual Grounding](https://arxiv.org/pdf/2105.11450.pdf)

by [Zhengyuan Yang](https://zhengyuan.info), [Songyang Zhang](https://sy-zhang.github.io/), [Liwei Wang](https://lwwangcse.github.io/), and [Jiebo Luo](http://cs.rochester.edu/u/jluo)

IEEE International Conference on Computer Vision (ICCV), 2021, Oral


## Introduction
We propose 2D Semantics Assisted Training (SAT) that assists 3D visual grounding with 2D semantics.
SAT helps 3D tasks with 2D semantics in training but does not require 2D inputs during inference.
For more details, please refer to our
[paper](https://arxiv.org/pdf/2105.11450.pdf).

<!-- Note: This codebase is still in beta release. We are continue organizing the repo and completing the doumentations. Meanwhile, please feel free to contact me regarding any issue and request for clarification. -->

<p align="center">
  <img src="https://zyang-ur.github.io//SAT/intro.jpg" width="75%"/>
</p>

## Citation

    @inproceedings{yang2021sat,
      title={SAT: 2D Semantics Assisted Training for 3D Visual Grounding},
      author={Yang, Zhengyuan and Zhang, Songyang and Wang, Liwei and Luo, Jiebo},
      booktitle={ICCV},
      year={2021}
    }

## Prerequisites
* Python 3.6.9 (e.g., conda create -n sat_env python=3.6.9 cudatoolkit=10.0)
* Pytorch 1.2.0 (e.g., conda install pytorch==1.2.0 cudatoolkit=10.0 -c pytorch)
* Install other common packages (numpy, [pytorch_transformers](https://pypi.org/project/pytorch-transformers/), etc.)
* Please refer to ``setup.py`` (From [ReferIt3D](https://github.com/referit3d/referit3d)).

## Installation

- Clone the repository

    ```
    git clone https://github.com/zyang-ur/SAT.git
    cd SAT
    pip install -e .
    ```

- To use a PointNet++ visual-encoder you need to compile its CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```Note: To do this compilation also need: gcc5.4 or later.```
    ```
    cd external_tools/pointnet2
    python setup.py install
    ```

## Data
### ScanNet
First you should download the train/val scans of ScanNet if you do not have them locally. Please refer to the [instructions from referit3d](referit3d/data/scannet/README.md) for more details. The output is the ``scanfile`` ``keep_all_points_00_view_with_global_scan_alignment.pkl/keep_all_points_with_global_scan_alignment.pkl``.

### Ref3D Linguistic Data
You can dowload the Nr3D and Sr3D/Sr3D+ from [Referit3D](https://github.com/referit3d/referit3d#our-linguistic-data), and send the file path to ``referit3D-file``.

### SAT Processed 2D Features
You can download the processed 2D object image features from [here](https://drive.google.com/file/d/1X_a9jWWNNkBGs49A3j4OKW4iz-gytnJA/view?usp=sharing). The cached feature should be placed under the ``referit3d/data`` folder, or match the cache path in the [dataloader](https://github.com/zyang-ur/SAT/blob/main/referit3d/in_out/pt_datasets/listening_dataset.py#L142). The feature extraction code will be cleaned and released in the future. Meanwhile, feel free to contact [me](zhengyuan.yang@gmail.com) if you need that before the official release.

## Training
Please reference the following example command on Nr3D. Feel free to change the parameters. Please reference [arguments](referit3d/in_out/arguments.py) for valid options.
  ```
  cd referit3d/scripts
  scanfile=keep_all_points_00_view_with_global_scan_alignment.pkl ## keep_all_points_with_global_scan_alignment if include Sr3D
  python train_referit3d.py --patience 100 --max-train-epochs 100 --init-lr 1e-4 --batch-size 16 --transformer --model mmt_referIt3DNet -scannet-file $scanfile -referit3D-file $nr3dfile_csv --log-dir log/$exp_id --n-workers 2 --gpu 0 --unit-sphere-norm True --feat2d clsvecROI --context_2d unaligned --mmt_mask train2d --warmup
  ```

## Evaluation
Please find the pretrained models [here](https://drive.google.com/drive/folders/14VZQHu38mZ0aoLbBAXM-_LHCMgBktH_Q?usp=sharing) (clsvecROI on Nr3D).
A known issue [here](https://github.com/zyang-ur/SAT/blob/main/referit3d/scripts/train_referit3d.py#L12).
  ```
  cd referit3d/scripts
  python train_referit3d.py --transformer --model mmt_referIt3DNet -scannet-file $scanfile -referit3D-file $nr3dfile --log-dir log/$exp_id --n-workers 2 --gpu $gpu --unit-sphere-norm True --feat2d clsvecROI --mode evaluate --pretrain-path $pretrain_path/best_model.pth
  ```

## Credits
The project is built based on the following repository:
* [ReferIt3D](https://github.com/referit3d/referit3d).

Part of the code or models are from [ScanRef](https://github.com/daveredrum/ScanRefer), [MMF](https://github.com/facebookresearch/mmf), [TAP](https://github.com/microsoft/TAP), and [ViLBERT](https://github.com/jiasenlu/vilbert_beta).