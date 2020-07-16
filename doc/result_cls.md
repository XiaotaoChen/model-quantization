
### Classification

Refer [classification.md](./classification.md) for detailed instructions.

Both the Top-1(\%) from original paper and the reproduction are listed. Corresponding training and testing configurations can be found in the `config` folder. Selected experiment results are listed. Users are encouraged to try different configurations to implement their own targets.

Note that the performance among different methods is obtained based on different training hyper-parameters. The accuracy in the table will not be the evidence of superior of one algorithm over another. Training hyper-parameters and tricks (such as `weight normalization`) play a considerable role on improving the performance. See the summary of my experience on training quantization networks in [experience.md](./experience.md).

We provide pretrained models in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing). Report missing pretrained model files if you cannot find it. 

Dataset | Method | Model | A/W | Reported | Top-1  | Flags | Config
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
imagenet | - | ResNet-18 | 32/32 | - | 70.1 | PreBN,bacs | [File](../config/config.dorefa.eval.imagenet.fp.resnet18)
imagenet | - | Torch-R18 | 32/32 | 69.8 | 70.1 | Pytorch-official |[File](../config/config.dorefa.eval.imagenet.fp.torch-resnet18)
imagenet | Fixup | ResNet-18 | 32/32 | - | 69.0 | fixup,cbsa,mixup=0.7 | [File](../config/)
imagenet | Fixup | ResNet-50 | 32/32 | - | 75.9 | fixup,cbsa,mixup=0.7 | [File](../config/config.fixup.eval.imagenet.fp.resnet50)
imagenet | TResnet | ResNet-18 | 32/32 | 70.1 | 68.7 | PreBN,bacs,TResNetStem | [File](../config/config.TResNet.eval.dali.fp.resnet18)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 64.9 | PreBN,bacs, ep120 (old)  | [File](../config/config.lq-net.eval.dali.2bit.resnet18)
imagenet | LQ-net | ResNet-18 | 2/2 | - | 65.9 | PreBN,bacs,fm_qg=8, ep120 (old) | [File](../config/config.lq-net.eval.dali.2bit.resnet18-fg8)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.7 | PreBN,bacs, ep120 | [File](../config/config.lq-net.finetune.dali.2bit.resnet18-baseline-sgdr)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.3 | PreBN,bacs,wt_mean-var, ep40
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.6 | PreBN,bacs,wt_mean-var, ep120 | [File](../config/config.lq-net.finetune.dali.2bit.resnet18-wt-norm)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.4 | PreBN,bacs,wt_mean-var,wt_gq=1, ep120
imagenet | LSQ | Torch-R18 | 2/2 | 67.6 | 67.3 | vanilla resnet(paper use pre act) | [File](../config/config.lsq.eval.imagenet.2bit.torch-resnet18)
imagenet | Dorefa-Net | ResNet-18 | 2/2 | - | 64.1 | PreBN,bacs  | [File](../config/config.dorefa.eval.imagenet.2bit.resnet18)
imagenet | Group-Net | ResNet-18 | 1/1 | - | 63.9 | cabs,bireal,base=5,without-softgate | [File](../config/config.group-net.eval.imagenet.bin.resnet18.base5.cabs)
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 52.0 | cbsa,fm_triangle,wt_pass,No-ReLU | [File](../config/config.xnor.train-strach.dali.bin.resnet18-triangle-pass)
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 50.5 | cbsa,fm_STE,wt_pass,No-ReLU
imagenet | LSQ | Torch-R18 | 1/1 | - | 58.5 | ReLU,wt-var-mean,wtg=1


`Torch-Rxx` indicates the ResNet architecture from Pytorch (so-called vanilla structure). `ResNet-xx` represents  the variants of ResNet. Minor differences are observed from different implementation from other projects. We provide flexible structure control to build compatibility of those projects. See [resnet.md](./resnet.md) for the architecture description and [classification.md](./classification.md) for how to control the choice by different configuration.

Explanations on some flags:

- cbsa / bacs:
  The resnet conv seq
  
- wt_var-mean:
  apply weight normalization (type `var-mean`) on the weight
  
- ep40 / ep120:
  total epoch of 40 / 120 in the training
  
- fm_qg/ wt_qg:
  quantization group
  
- real shortcut / real-skip: the downsample layer is kept in full precision. Other wise the shortcut is quantized (eg. `2bit shortcut`)

- old:  indicating that better results are obtained but still not updated in the table.

