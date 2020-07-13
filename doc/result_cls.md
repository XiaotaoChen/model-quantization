
### Classification

Refer [classification.md](./classification.md) for detailed instructions.

Both the Top-1(\%) from original paper and the reproduction are listed. Corresponding training and testing configruations can be found in the `config` folder. Selected experiement results are listed. Users are encouraged to try different configurations to implement their own targets.

Note that the performance among different methods is obtained based on different training hyper-parameters. The accuracy in the table will not be the evidence of superior of one algorithm over another. Training hyper-parameters and tricks (such as `weight normalization`) play a considerable role on improving the performance. See the summary of my experience on training quantization networks in [experience.md](./experience.md).

We provide some pretrained models in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)

Dataset | Method | Model | A/W | Reported | Top-1  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:
imagenet | - | ResNet-18 | 32/32 | - | 70.1 | PreBN,bacs 
imagenet | - | Torch-R18 | 32/32 | 69.8 | 70.1 | Pytorch-official
imagenet | Fixup | ResNet-18 | 32/32 | - | 69.0 | fixup,cbsa,mixup=0.7
imagenet | Fixup | ResNet-50 | 32/32 | - | 75.9 | fixup,cbsa,mixup=0.7
imagenet | TResnet | ResNet-18 | 32/32 | 70.1 | 68.7 | PreBN,bacs,TResNetStem
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 64.9 | PreBN,bacs, ep120 (old)
imagenet | LQ-net | ResNet-18 | 2/2 | - | 65.9 | PreBN,bacs,fm_qg=8, ep120 (old)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.7 | PreBN,bacs, ep120
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.3 | PreBN,bacs,wt_mean-var, ep40
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.6 | PreBN,bacs,wt_mean-var, ep120
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.4 | PreBN,bacs,wt_mean-var,wt_gq=1, ep120
imagenet | LSQ | Torch-R18 | 2/2 | 67.6 | 67.3 | vanilla resnet(paper use pre act)
imagenet | Dorefa-Net | ResNet-18 | 2/2 | - | 64.1 | PreBN,bacs
imagenet | Group-Net | ResNet-18 | 1/1 | - | 63.9 | cabs,bireal,base=5,without-softgate
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 52.0 | cbsa,fm_triangle,wt_pass,No-ReLU
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 50.5 | cbsa,fm_STE,wt_pass,No-ReLU
imagenet | LSQ | Torch-R18 | 1/1 | - | 58.5 | ReLU,wt-var-mean,wtg=1
cifar100 |  - | ResNet-20 | 32/32 | - | 66.56 | cbsa, ldn, order c
cifar100 |  - | ResNet-20 | 32/32 | - | 75.13 | cbsa, ldn, order cb
cifar100 |  - | ResNet-20 | 32/32 | - | 71.82 | cbsa, ldn, order ca
cifar100 |  - | ResNet-20 | 32/32 | - | 66.90 | cbsa, ldn, order cba
cifar100 |  - | ResNet-20 | 32/32 | - | 66.88 | cbsa, ldn, baseline
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 63.51 | cbsa, ldq, order c, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 73.84 | cbsa, ldq, order cb, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 69.98 | cbsa, ldq, order ca, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 58.94 | cbsa, ldq, order cba, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 63.25 | cbsa, ldq, baseline, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 73.59 | cbsa, ldq, order cb, 2bit shortcut

`Torch-Rxx` indicates the ResNet architecture from Pytorch (so-called vanilla structure). `ResNet-xx` represnets the variants of ResNet. Minior differences are observed from different implementation from other projects. We provide flexible structure control to build compatibility of those projects. See [resnet.md](./resnet.md) for the architecture description and [classification.md](./classification.md) for how to control the choice by different configuration.

Explanations on some flags:

- cbsa / bacs:
  The resnet conv seq
  
- wt_var-mean:
  apply weight normalization (type `var-mean`) on the weight
  
- ep40 / ep120:
  training total epoch of 40 / 120
  
- fm_qg/ wt_qg:
  quantization group
  
- real shortcut / real-skip: the downsample layer is kept in full precision. Other wise the shortcut is quantized (eg. `2bit shortcut`)

- ldq: Lossless downsample quantization. ldn: Lossless downsample network.


