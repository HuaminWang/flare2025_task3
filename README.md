# Solution for Flare2025 Task3

This repository is the official implementation of [Unsupervised Domain Adaptation for Cross-modality Abdominal Organ Segmentation via Organ Attention Style Transfer and Dual-stage Pseudo Label Filtering](https://openreview.net/forum?id=dr1h6OOthW) of Team hilab on FLARE 2025 task 3 challenge.


## Style Translation
We use `OrganAttenCycleGAN` to translate images from CTs to MR/PETs while preserving anatomical structures.  
The preprocessing pipeline for converting 3D images to 2D slices before image translation can be found in the folder `flare25_styletranslation/data`, and the training script is located in `OrganAttenCycleGAN`.

## Segmentation
The dual-stage segmentation framework is based on the [previous winning solution](https://github.com/TJUQiangChen/FLARE24-task3).  
We introduce a dual-stage pseudo-label filtering implementation, which can be found in `lab_filter.ipynb`.



## Results

Our method achieves the following performance on [FLARE2025](https://www.codabench.org/competitions/2296/)


| Dataset Name       |   MR DSC(%)   | MR NSD(%)   | PET DSC(%)  | PET NSD(%)  |
|--------------------|:-------------:|:-----------:|:-----------:|:-----------:|
| Validation Dataset | 81.21%        | 88.54%      | 81.43%      | 71.94%      |
| Test Dataset       | (?)           | (?)         | (?)         | (?)         |


## Docker
A `Dockerfile` is provided, and the official Docker image is available on [Hugging Face](https://huggingface.co/huaminwang/flare2025_task3).
```
docker load -i hilab.tar.gz
docker container run --gpus "device=1" -m 28G --name hilab --rm -v $PWD/FLARE_Test/:/workspace/inputs/ -v $PWD/hilab_outputs/:/workspace/outputs/ hilab:latest /bin/bash -c "sh predict.sh"
```


## Acknowledgement
We would like to thank the contributors of the [FLARE25 dataset](https://www.codabench.org/competitions/2296/) and the authors of the [Champion Solution for FLARE24-Task3](https://github.com/TJUQiangChen/FLARE24-task3) for their valuable resources and efforts.




## Reference
```
@inproceedings{wang2025unsupervised,
  title={Unsupervised Domain Adaptation for Cross-modality Abdominal Organ Segmentation via Organ Attention Style Transfer and Dual-stage Pseudo Label Filtering},
  author={Wang, Huamin and Wu, Jianghao and Wang, Guotai and Zhou, Xianhao and He, Jinlong},
  booktitle={MICCAI 2025 FLARE Challenge}
}

@incollection{li20243d,
  title={A 3d unsupervised domain adaptation framework combining style translation and self-training for abdominal organs segmentation},
  author={Li, Jiaxi and Chen, Qiang and Ding, Haoyu and Liu, Hongying and Wan, Liang},
  booktitle={MICCAI Challenge on Fast and Low-Resource Semi-supervised Abdominal Organ Segmentation},
  pages={209--224},
  year={2024},
  publisher={Springer}
}

@incollection{wu2024unsupervised,
  title={Unsupervised domain adaptation for abdominal organ segmentation using pseudo labels and organ attention cyclegan},
  author={Wu, Jianghao and Zhang, Guoning and Qi, Xiaoran and Wang, Huamin and Liu, Xinya and Wang, Guotai},
  booktitle={MICCAI Challenge on Fast and Low-Resource Semi-supervised Abdominal Organ Segmentation},
  pages={225--242},
  year={2024},
  publisher={Springer}
}
```
