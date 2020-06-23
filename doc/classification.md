# Quantization for image classification

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


## Training script options

### Weight decay

Three major related options.

1. `--wd` set the default L2 weight decay value.

2. Weight decay is originally proposed to avoid ovrefit for the large amount of paramters. For some small tensors, for example the parameters in BatchNorm layer (as well as custom defined quantization parameter, such as clip-value), weight decay is advocated to be zero. `--decay_small` is for whether decay those small tensor or not.

3. `--custom_decay_list` and `--custom_decay` are combined for specific custom decay value to certain parameters. For example, in PACT, the clip_boudary can own its independent weight decay for regularition. The combination filter paramter name according to `--custom_decay_list` and assgin the weight decay to `--custom_decay`.


### Learning rate

