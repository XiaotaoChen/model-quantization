
# EasyQuantization: A flexible and unified framework to make the model quantization easier.

This project collects model quantization algorithms in the publication.

## Dashboard

The dashboard collects the perfromance of quantization algoirthms for different architectures. Both the Top-1(\%) from original paper and the reproduction are listed. Corresponding training and testing configruations can be found in the `config` folder.

Note that the performance among different methods is obtained based on different training hyper-parameters. The accuracy in the table will not be the evidence of superior of one algorithms over another. Training hyper-parameters and tricks such as `weight normalization` play a considerable role on improving the performance. See my experience summary of training quantization networks in [experience.md](./doc/experience.md).

Dataset | Method | Model | A/W | Reported | Top-1  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:
imagenet | - | ResNet-18 | 32/32 | - | 70.1 | PreBN,bacs 
imagenet | - | Torch-R18 | 32/32 | 69.8 | 70.1 | Pytorch-official
imagenet | Fixup | ResNet-18 | 32/32 | - | 69.0 | fixup,cbsa,mixup=0.7
imagenet | Fixup | ResNet-50 | 32/32 | - | 75.9 | fixup,cbsa,mixup=0.7
imagenet | TResnet | ResNet-18 | 32/32 | 70.1 | 68.7 | PreBN,bacs,TResNetStem
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 64.9 | PreBN,bacs 
imagenet | LQ-net | ResNet-18 | 2/2 | - | 65.9 | PreBN,bacs,fm-qg=8
imagenet | LSQ | Torch-R18 | 2/2 | 67.6 | 67.3 | vanilla resnet(paper use pre act)
imagenet | Dorefa-Net | ResNet-18 | 2/2 | - | 64.1 | PreBN,bacs
imagenet | Group-Net | ResNet-18 | 1/1 | - | 63.9 | cabs,bireal,base=5,without-softgate

`Torch-Rxx` indicates the ResNet architecture from Pytorch. `ResNet-xx` represnets the variants of ResNet. Minior differences can be found in the structure.


## Update History

## Install

1. clone the repo (change the `FASTDIR` as perfered):
```
export FASTDIR=/workspace
cd $FASTDIR/git/
git clone https://github.com/blueardour/model-quantization
git clone https://github.com/blueardour/pytorch-utils
cd model-quantization
ln -s ../pytorch-utils utils
# log and weight folders (optional, if symbol link not created, the script will create these folders under the project path)
mkdir -p /data/pretrained/pytorch/model-quantization/{exp,weights}
ln -s /data/pretrained/pytorch/model-quantization/exp .
ln -s /data/pretrained/pytorch/model-quantization/weights .
```

2. install prerequisite packages
```
cd $FASTDIR/git/model-quantization
# python 3 is required
pip install -r requirement.txt
```

3. Install Nvidia Image preprocess packages and mix precision training packages (optional)

[Nvidia Dali](https://github.com/NVIDIA/DALI) 

[Nvidia Mix precision training package](https://github.com/NVIDIA/apex)

## Dataset

This repo support the imagenet dataset and cifar dataset. 
Create necessary folder and prepare the datasets. Example:

```
# dataset
mkdir -p /data/cifar
mkdir -p /data/imagenet
# download imagnet and move the train and evaluation data in in /data/imagenet/{train,val}, respectively.
# cifar dataset can be downloaded on the fly
```

## Quick Start

Both training and testing employ the `train.sh` script. Directly call the `main.py` is also possible.

```
bash train.sh config.xxxx
```

`config.xxxx` is the configuration file, which contains network architecture, quantization related and training related parameters. For more about the supported options, refer [config.md](./doc/config.md) and the `config` subfolder.

Sometime the training is time-consuming. `start_on_terminate.sh` can be used to wait the process to terminate and start another round of training.

```
# wait in a screen shell
screen -S next-round
bash start_on_terminate.sh [current training thread pid] [next round config.xxxx]
# Ctl+A D to detach screen to backend
```

Besides, `tools.py` provides many useful functions for debug / verbose / model convert. Refer [tools.md](./doc/tools.md) for detailed usage.

## Algorithms

1. LQnet:

Please cite if you use this method

```BibTeX
@inproceedings{zhang2018lq,
  title={LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks},
    author={Zhang, Dongqing and Yang, Jiaolong and Ye, Dongqiangzi and Hua, Gang},
      booktitle=ECCV,
        year={2018}
}
```

## Contributors

Current Contributors:

- [Blueardour](https://github.com/blueardour)
- [bohanzhuang](https://github.com/bohanzhuang)
- [liujingcs](https://github.com/liujingcs)
- [liuchunlei0430](https://github.com/liuchunlei0430)


To contribute, PR is appreciated and it is also possible to contact me by email: blueardour@gamil.com

## License


