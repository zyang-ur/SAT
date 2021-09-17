# SAT: 2D Semantics Assisted Training for 3D Visual Grounding
[SAT: 2D Semantics Assisted Training for 3D Visual Grounding](https://arxiv.org/pdf/2105.11450.pdf)

by [Zhengyuan Yang](zhengyuan.info), [Songyang Zhang](https://sy-zhang.github.io/), [Liwei Wang](https://lwwangcse.github.io/), and [Jiebo Luo](http://cs.rochester.edu/u/jluo)

IEEE International Conference on Computer Vision (ICCV), 2021, Oral


### Introduction
We propose 2D Semantics Assisted Training (SAT) that assists 3D visual grounding with 2D semantics.
SAT helps 3D tasks with 2D semantics in training but does not require 2D inputs during inference.
For more details, please refer to our
[paper](https://arxiv.org/pdf/2105.11450.pdf).

<!-- Note: This codebase is still in beta release. We are continue organizing the repo and completing the doumentations. Meanwhile, please feel free to contact me regarding any issue and request for clarification. -->

<p align="center">
  <img src="https://zyang-ur.github.io//SAT/intro.jpg" width="75%"/>
</p>

### Citation

    @inproceedings{yang2021sat,
      title={SAT: 2D Semantics Assisted Training for 3D Visual Grounding},
      author={Yang, Zhengyuan and Zhang, Songyang and Wang, Liwei and Luo, Jiebo},
      booktitle={ICCV},
      year={2021}
    }

### Prerequisites
* Python 3.6.9
* Pytorch 1.4.0
* Please refer to ``setup.py`` (Thanks to [ReferIt3D](https://github.com/referit3d/referit3d)). Or using

  ```
  python setup.py develop
  ```

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


### Credits
The project is built based on the following repository:
* [ReferIt3D](https://github.com/referit3d/referit3d).