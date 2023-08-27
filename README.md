# [ICCV 2021] GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning
[Haipeng Li](https://lhaippp.github.io/), [Kunming Luo](https://coolbeam.github.io/index.html), [Shuaicheng Liu](http://www.liushuaicheng.org/),

This is the official implementation of our ICCV2021 paper [GyroFlow](https://openaccess.thecvf.com/content/ICCV2021/html/Li_GyroFlow_Gyroscope-Guided_Unsupervised_Optical_Flow_Learning_ICCV_2021_paper.html).

Our presentation video: [[Youtube](https://www.youtube.com/watch?v=6gh40PyWdHM)][[Bilibili](https://www.bilibili.com/video/BV1Tr4y127kd/)].

2023-07: Try our Journal Extended Version [GyroFlow+: Gyroscope-Guided Unsupervised Deep Homography and Optical Flow Learning](https://github.com/lhaippp/GyroFlowPlus)

## Dependencies

* Requirements please refer to`requirements.txt`.

## Data Preparation

We provide a toy demo to illustrate the process of converting gyroscope readings (i.e., angular velocity in row, pith, yaw) into an homography at [gyro-video-stabilization](https://github.com/lhaippp/gyro-video-stabilization), welcme to play with it

### GOF-Train

2021.11.15: We release the GOF_Train that contains 2000 samples.

2023.07.10: We release the GHOF_Train that contains 9900 samples.

The download link is [GoogleDrive](https://drive.google.com/file/d/1duHQBIWLOPHd5LxBBLpsy6-3FAg_kqNp/view?usp=sharing). Put the data into `./dataset/GHOF_Train`, and the contents of directories are as follows:

```
./dataset/GOF_Train
├── sample_0
│   ├── img1.png
│   ├── img2.png
│   ├── gyro_homo.npy
├── sample_1
│   ├── img1.png
│   ├── img2.png
│   ├── gyro_homo.npy
.....................
├── sample_9900
│   ├── img1.png
│   ├── img2.png
│   ├── gyro_homo.npy

```
### GHOF
2023.07.10: We release the GHOF_Clean&Final that contains 5 categories, as the benchmark is changed, we thus update the metrics.

The pretrained model can be found at [GoogleDrive](https://drive.google.com/file/d/1dE7jKZS6RJNLWSrDY0q_pIJUFeo4JTqn/view?usp=sharing). Move the model to `./experiments/demo_experiment/exp_2/test_model_best.pth`.

| BMK |  AVG   |  RE  |  FOG  |  DARK  |  RAIN  |  SNOW  |
|  ----  |  ----  | ----  |  ----  | ----  |  ----  | ----  |
|  Clean+Final  | 1.23  | 1.10 |  1.10  | 2.37  |  0.52  | 1.07  |

### GHOF-Clean

For quantitative evaluation, including input frames and the corresponding gyro readings, a ground-truth optical flow is required for each pair.

The download link is [GoogleDrive](). Move the file to `./dataset/GHOF_Clean.npy`.

| BMK |  AVG   |  RE  |  FOG  |  DARK  |  RAIN  |  SNOW  |
|  ----  |  ----  | ----  |  ----  | ----  |  ----  | ----  |
|  Clean  | 1.08  | 0.88 |  0.90  | 2.20  |  0.44  | 0.83  |

### GHOF-Final

The most difficult cases are collected in GOF-Final.

The download link is [GoogleDrive](). Move the file to `./dataset/GHOF_Final.npy`.

| BMK |  AVG   |  RE  |  FOG  |  DARK  |  RAIN  |  SNOW  |
|  ----  |  ----  | ----  |  ----  | ----  |  ----  | ----  |
|  Final  | 1.36  | 1.31 |  1.30  | 2.55  |  0.59  | 1.25  |
## Training and Evaluation

### Training

To train the model, you can just run:

```
python train.py --model_dir experiments
```

### Evaluation

Load the pretrained checkpoint and run:

```
python test.py --model_dir experiments/demo_experiment/exp_2 --restore_file experiments/demo_experiment/exp_2/test_model_best.pth
```

## Citation

If you think this work is useful for your research, please kindly cite:

```
@InProceedings{Li_2021_ICCV,
    author    = {Li, Haipeng and Luo, Kunming and Liu, Shuaicheng},
    title     = {GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12869-12878}
}
```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [ARFlow](https://github.com/lliuz/ARFlow)
* [UpFlow](https://github.com/coolbeam/UPFlow_pytorch)
* [RAFT](https://github.com/princeton-vl/RAFT)
* [DeepOIS](https://github.com/lhaippp/DeepOIS)

We thank the respective authors for open sourcing their methods.
