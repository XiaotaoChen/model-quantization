
# EasyQuantization: A flexible and unified framework to make the model quantization easier.

This project collects model quantization algorithms in the publication.

## Dashboard

The dashboard collects the perfromance of quantization algoirthms for different architectures. Computer vision tasks such as image classification, detetion, segmentation are inlcuded. Note that detection and segmentation might employ a seperate repo with this one as a submodule.

### Classification

Refer [classification.md](./doc/classification.md) for detailed instructions.

Both the Top-1(\%) from original paper and the reproduction are listed. Corresponding training and testing configruations can be found in the `config` folder. Limited among of experiement results are listed. Users are encouraged to try different configurations to implement their own targets.

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
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 52.0 | cbsa,fm_triangle,wt_pass,No-ReLU
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 50.5 | cbsa,fm_STE,wt_pass,No-ReLU

`Torch-Rxx` indicates the ResNet architecture from Pytorch (so-called vanilla structure). `ResNet-xx` represnets the variants of ResNet. Minior differences can be found in the structure. See [resnet.md](./doc/resnet.md) for the architecture description and  [classification.md](./doc/classification.md) for how to control the choice by different configuration.

### Detection

Selected results are listed in the following table. More data and detailed instructions can be found in [detection.md](./doc/detection.md)

Dataset | Task Method | Quantization method | Model | A/W | Reported | Top-1  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
COCO | Retina-Net | - | Torch-18 | 32/32 | - | - | 

## Update History

- Detection (preparing)

- 2020.06.23 Add classification quantization



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


To contribute, PR is appreciated and suggestions are welcome to discuss with me by email: blueardour@gamil.com

## License


