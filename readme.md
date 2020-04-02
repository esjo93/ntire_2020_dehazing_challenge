# NTIRE 2020 NonHomogeneous Dehazing Challenge: UNIST VIP Lab
## Introduction
This is our project repository for CVPR 2020 workshop.

**"Physical Encoder-Decoder Network for Image Dehazing"**

## Network Architecture
![architecture](./figure/architecture.png)

## Dataset Preparation
You can download **NTIRE 2020 NonHomogeneous Dehazing Challenge** dataset after participating the challenge in the following link:
[https://competitions.codalab.org/competitions/22236](https://competitions.codalab.org/competitions/22236)

Your dataset directory should be composed of three directories like following:
```bash
dataset_directory
|-- train
|   |-- HAZY
|   |   |-- 01
|   |   |-- 02
|   |   `-- ...
|   `-- GT
|       |-- 01
|       |-- 02
|       `-- ...
|-- val
|   |-- HAZY
|   |   `-- ...
|   `-- GT
|       `-- ...
`-- test
    `-- HAZY
        `-- ...
```

## Train
You can start training your model by following:
```
$ python main.py train
Additional arguments:
    --data-dir: Dataset directory
    --batch-size: Training batch size
    --epochs: The number of total epochs
    --lr: Initial learning rate
    --step: Step size for learning rate decay
    --weight-decay: Weight decay factor
    --crop-size: Random crop size for training
```


## Test
You can test your pretrained model by following:
```
$ python main.py test -d [data path] --resume [pretrained model path] --phase test --batch-size 1
```

Download pretrained model: [[download](https://drive.google.com/open?id=1wLYJWlMtSOmcU_GP-4TMcV9uHDX8TLlP)]

## Results
| Metrics | Test Scores (#51~55)
|:----:|:----:|
| PSNR | 18.77 |
| SSIM | 0.54 |
| Run time[s] per img. | 0.04 | |

![results](./figure/results.PNG)
