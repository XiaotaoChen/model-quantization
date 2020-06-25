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
For Pytorch, the quantization project for classification task has no stricted requirement. Version above `Torch 1.0` should all work fine. However, seems the [detection](./detection.md) project requires a higher version pytorch. They currently require `Torch 1.4`+.

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

`config.xxxx` is the configuration file, which contains network architecture, quantization related and training related parameters. For more about the supported options, refer below [Training script options](./classification.md#Training-script-options) and [config.md](./config.md). Also refer the examples in `config` subfolder.

Sometime the training is time-consuming. `start_on_terminate.sh` can be used to wait the process to terminate and start another round of training.

```
# wait in a screen shell
screen -S next-round
bash start_on_terminate.sh [current training thread pid] [next round config.xxxx]
# Ctl+A D to detach screen to backend
```

Besides, `tools.py` provides many useful functions for debug / verbose / model convert. Refer [tools.md](./tools.md) for detailed usage.


## Training script options

- Option parsing

Common options are parsed in `util/config.py`. Quantization related options are separated in the `main.py`.

- Keyword

The `--keyword` option is one of most important varibale to control the model architecture and quantization algorithm choice.

Currently support quantization algorithm choice by add the following items in the `keyword`:

a. `lq` for Lq-net

b. `pact` for PACT

c. `dorefa` for dorefa-net. Besides, additional keyword of `lsq` for learned step size, `non-uniform` for FATNN.

d. `xnor` for xnor-net

Structure control keyword:

a. `origin` to choose whether the bi-real skip connection is perfered.

b. `bacs` or `cbas` so on indicate the layer order in a resnet block. For example, `bacs` is a kind of pre-activation structure, representing in a resnet block, first normalization layer, then activation layer, then convolution layer and last skip connection layer. For pre-activation structure, `preBN` is required for the first resnet block.  Refer [resnet.md](./resnet.md) for more information.

c. By default all layers except the first and last layer are quantized, `real_skip` can be added to keep the skip connection layers in resnet to full precision. Widely used in Xnor-net and Bi-Real net.

d. The normalization layer and activation layer, we also provide some `keyword` for different variants. Refer `model/layer.py` for the detail. 

e. I think it is an error if padding the feature map with 0 after quantization, specially in BNN. From my perspective, the strategy makes BNNs to become TNNs. Thus, I advocate to pad the feature map with zero first and then go through the quanzation step. To keep compatible with the publication as well as provide a revision method, `padding_after_quant` is supplied the order between padding and quantization. Refer line 445 in `model/quant.py` for the implementation.

Customed `keyword` is suported and can be easily realized according the user's own desire. The options can be combined to build up different variant architecuture. Reference cases can be found in the `config` subfolder.

- Activation and weight quantization options

The script provides indepdent configration for the activation and weight, respectively. Options such as `xx_bit`, `xx_level`, `xx_enable`, `xx_half_range` are easy to understand (`xx` is `fm` for activation or `wt` for weight ). We here explain more about other advanced options. 

1. `xx_quant_group` indicates the group amount for the quantization parameter along the channel dimension.

2. `xx_adaptive` in most cases, inidicates the additonal normalization operation which shows great potential to increase the performance.

3. `xx_grad_type` define custom gradident boost method. As generally, the quantization step is not differentiable, techniques such as the STE are used to approximate the gradient. Other types of approximation exist. Besides, in some publication, it is advocated to add some scale coefficient to the gradient in order to stabilize the training.

- Weight decay

Three major related options.

1. `--wd` set the default L2 weight decay value.

2. Weight decay is originally proposed to avoid ovrefit for the large amount of paramters. For some small tensors, for example the parameters in BatchNorm layer (as well as custom defined quantization parameter, such as clip-value), weight decay is advocated to be zero. `--decay_small` is for whether decay those small tensor or not.

3. `--custom_decay_list` and `--custom_decay` are combined for specific custom decay value to certain parameters. For example, in PACT, the clip_boudary can own its independent weight decay for regularition. The combination filter paramter name according to `--custom_decay_list` and assgin the weight decay to `--custom_decay`.


- Learning rate

1. multi-step decay

2. ploy decay

3. sgdr (with restart)

4. `--custom_lr_list` and `--custom_lr` are provided simiarly with beforemetioned weight decay to specific custom learning rate for certain paramaters.

- mix precision training

1. FP32

2. FP16, recommmond `O1` level.
