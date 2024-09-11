# MPO

This is the implementation of Radiology Report Generation via Multi-objective Preference Optimization.

## Requirements

- pip install -r requirement.txt

## Download MPO
<<<<<<< HEAD
You can download the models we pre-trained model for each dataset from [https://pan.baidu.com/s/1U11ptNWzbyYF2pe8Btr86g]() 提取码：2nq7
=======

You can download the models we pre-trained model for each dataset from []()
>>>>>>> 9d8f35ce96ecd45a646497172427d0d6c3ef0b6b

you can download the best models for each dateset from [https://pan.baidu.com/s/19JWHjHTSKHLmruo8CnwLyg]() 提取码：52tc

## Datasets

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and place the files in the specified path within scripts/iu-xray/run_rl.sh."

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then the specified path in `scripts/mimic-cxr/run_rl.sh`. You can apply the dataset [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

## Train

Run `bash scripts/iu_xray/run_rl.sh` to train a model on the IU X-Ray data.

Run `bash scripts/mimic-cxr/run_rl.sh` to train a model on the MIMIC-CXR data.

You can download the pre-trained model for CheXbert from here: [Chexbert](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9). Then place it in the `MPO_IU\MPO_TRAIN\CheXbert\checkpoint` directory.

For using RadGraph, you can refer to the following link: [RadGraph](https://github.com/hlk-1135/RadGraph). The specific model checkpoint can be downloaded from here: [model_checkpoint](https://physionet.org/content/radgraph/1.0.0/models/model_checkpoint/#files-panel). Place the related files in my `MPO_IU\MPO_TRAIN\RadGraph` directory.

## Test

在PMO_TEST

Run `bash MPO/MPO_TEST/test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash MPO/MPO_TEST/test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.

