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

# create separate log and weight folders (optional, if symbol link not created, the script will create these folders under the project path)
#mkdir -p /data/pretrained/pytorch/model-quantization/{exp,weights}
#ln -s /data/pretrained/pytorch/model-quantization/exp .
#ln -s /data/pretrained/pytorch/model-quantization/weights .
```

2. install prerequisite packages
```
cd $FASTDIR/git/model-quantization
# python 3 is required
pip install -r requirement.txt
```
The quantization for classification task requires the pytorch version `1.3` or higher version. However, other tasks such as detection and segmentation require a higher version pytorch. `detectron2` currently require `Torch 1.4`+. Besides, the CUDA version on the machine is advised to keep same with the one compiling the pytorch.

3. Install Nvidia Image pre-process packages and mix precision training packages (optional, highly recommend)

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

## Pretrained model

Some of the quantization results are listed in [result_cls.md](./result_cls.md). We provide pretrained models in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)

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

***ALl complains about the following two warnings do not affect the project.***
```
Failing to import plugin, ModuleNotFoundError("No module named 'plugin'")
loading third party model failed cannot import name 'model_zoo' from 'third_party' (unknown location)
```


## Training script options

- Option parsing

  Common options are parsed in `util/config.py`. Quantization related options are separated in the `main.py`.

- Keyword (quantization method choosing)

  The `--keyword` option is one of most important variables to control the model architecture and quantization algorithm choice.

  Currently support quantization algorithm choice by add the following items in the `keyword`:

  a. `lq` for Lq-net
  
  b. `pact` for PACT
  
  c. `dorefa` for dorefa-net. Besides, additional keyword of `lsq` for learned step size, `non-uniform` for FATNN.
  
  d. `xnor` for xnor-net. if `gamma` is combined with the `xnor` in the keyword, a separated learnable scale coefficient is added (It namely becomes the XNor-net++).

- Keyword (structure control):

  The network  structure, of course, is firstly decided by the model architecture choosing (by `--arch` or `--model`). For ResNet, the official ResNet model is provided with `pytorch-resnetxx` and more flexible ResNet architecture can be realized by setting the `--arch` or `--model` with `resnetxx`. For the latter case, a lot of options can be combined to customize the network structure:

  a. `origin` exists / not exists in `keyword` is to choose whether the bi-real skip connection is preferred (Block-wise skip connection versus layer-wise skip connection).
  
  b. `bacs` or `cbas`, etc, indicate the layer order in a ResNet block. For example, `bacs` is a kind of pre-activation structure, representing in a resnet block, first normalization layer, then activation layer, then convolution layer and last skip connection layer. For pre-activation structure, `preBN` is required for the first resnet block.  Refer [resnet.md](./resnet.md) for more information.
  
  c. By default all layers except the first and last layer are quantized, `real_skip` can be added to keep the skip connection layers in resnet to full precision. Widely used in Xnor-net and Bi-Real net.
  
  d. The normalization layer and activation layer, we also provide some `keyword` for different variants. For example, `NRelU` means do not including ReLU activation in the network and `PRelU` indicates PReLU is employed. Refer `model/layer.py` for the detail. 
  
  e. Padding and quantization order. I think it is an error if padding the feature map with 0 after quantization, especially in BNN. From my perspective, the strategy makes BNNs to become TNNs. Thus, I advocate to pad the feature map with zero first and then go through the quantization step. To keep compatible with the publication as well as provide a revision method, `padding_after_quant` is supplied the order between padding and quantization. Refer line 445 in `model/quant.py` for the implementation.
  
  f. Skip connection realization. Two choices are provided. One is a avgpooling with stride followed by a conv1x1 with stride=1. The other is just one conv1x1 with stride as demanded. `singleconv` in `keyword` is used for the choice.
  
  g. `fixup` is used to enable the architecture in Fixup Initialization. 
  
  h. the option `base` which is a standalone option rather a word in the `keyword` list is used to realize the branch configuration in Group-Net
  
  Self-defined `keyword` is supported and can be easily realized according the user's own desire. As introduced above, the options can be combined to build up different variant architectures. Examples can be found in the `config` subfolder.

- Activation and weight quantization options

  The script provides independent configuration for the activation and weight, respectively. Options such as `xx_bit`, `xx_level`, `xx_enable`, `xx_half_range` are easy to understand (`xx` is `fm` for activation or `wt` for weight ). We here explain more about other advanced options. 
  
  1. `xx_quant_group` indicates the group amount for the quantization parameter along the channel dimension.
  
  2. `xx_adaptive` in most cases, indicates the additional normalization operation which shows great potential to increase the performance.
  
  3. `xx_grad_type` define custom gradient boost method. As generally, the quantization step is not differentiable, techniques such as the STE are used to approximate the gradient. Other types of approximation exist. Besides, in some publication, it is advocated to add some scale coefficient to the gradient in order to stabilize the training.

- Weight decay

  Three major related options.

  1. `--wd` set the default L2 weight decay value.
  
  2. Weight decay is originally proposed to avoid overfit for the large number of parameters. For some small tensors, for example the parameters in BatchNorm layer (as well as custom defined quantization parameter, such as clip-value), weight decay is advocated to be zero. `--decay_small` is for whether decay those small tensors or not.
  
  3. `--custom_decay_list` and `--custom_decay` are combined for specific custom decay value to certain parameters. For example, in PACT, the clip_boudary can own its independent weight decay for regulation. The combination filter parameter name according to `--custom_decay_list` and assign the weight decay to `--custom_decay`.


- Learning rate

  1. multi-step decay
  
  2. ploy decay
  
  3. sgdr (with restart)
  
  4. `--custom_lr_list` and `--custom_lr` are provided similarly with before mentioned weight decay to specific custom learning rate for certain parameters.

- mix precision training
  options `--fp16` and `--opt_level [O1]` are provided for mix precision traning.

  1. FP32
  
  2. FP16 with customed level, recommend `O1` level.
