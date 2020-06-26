# Quantization object detection/segmentation tasks

## Dashboard

Here lists selected experiment result. The performance is potentially being better if effort is paid on tuning. See [experience.md](experience.md) to communicate training skills with me.

Dataset | Task Method | Quantization method | Model | A/W | Reported | AP  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
COCO | Retina-Net | - | Torch-18 | 32/32 | - | 31.5 | 1x
COCO | Retina-Net | - | Torch-18 | 32/32 | - | 32.8 | 1x, FPN-BN,Head-GN
COCO | Retina-Net | - | Torch-18 | 32/32 | - | 33.0 | 1x, FPN-BN,Head-BN
COCO | Retina-Net | - | Torch-34 | 32/32 | - | 35.2 | 1x
COCO | Retina-Net | - | Torch-50 | 32/32 | - | 36.6 | 1x
COCO | Retina-Net | - | Torch-50 | 32/32 | - | 37.8 | 1x, FPN-BN,Head-BN
COCO | Retina-Net | - | MSRA-R50 | 32/32 | - | 36.4 | 1x
COCO | FCOS | - | MSRA-R50 | 32/32 | - | 38.6 | 1x
COCO | FCOS | - | Torch-50 | 32/32 | - | 38.4 | 1x
COCO | FCOS | - | Torch-50 | 32/32 | - | 38.5 | 1x,FPN-BN
COCO | FCOS | - | Torch-50 | 32/32 | - | 38.9 | 1x,FPN-BN,Head-BN
COCO | FCOS | - | Torch-34 | 32/32 | - | 37.3 | 1x
COCO | FCOS | - | Torch-18 | 32/32 | - | 32.2 | 1x
COCO | FCOS | - | Torch-18 | 32/32 | - | 33.4 | 1x,FPN-BN
COCO | FCOS | - | Torch-18 | 32/32 | - | 33.9 | 1x,FPN-BN, FP16
COCO | FCOS | - | Torch-18 | 32/32 | - | 33.9 | 1x,FPN-BN,Head-BN
COCO | FCOS | - | Torch-18 | 32/32 | - | 34.3 | 1x,FPN-SyncBN,Head-SyncBN
COCO | FCOS | Dorefa-Net | Torch-18 | 2/2 | - | 33.4 | 1x,FPN-BN, Quantize-Backbone
COCO | FCOS | Dorefa-Net | Torch-18 | 2/2 | - | 30.3 | 1x,FPN-BN, Quantize-All
COCO | FCOS | LQ-Net | Torch-18 | ter/ter | - | 32.6 | 1x,FPN-BN, Quantize-Backbone
COCO | FCOS | LQ-Net | Torch-18 | ter/ter | - | 26.2 | 1x,FPN-BN, Quantize-All


## Install

1. download the [custom detectron2](https://github.com/blueardour/detectron2) project. See what is modified below.

```
cd /workspace/git/
git clone https://github.com/blueardour/detectron2
# checkout the quantization branch
cd detectron2
git checkout quantization
```
Facebook detectron2 has not support for some works such as `FCOS` and `Blendmask`. Try the [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for more task support. Note, for the `aim-uofa/AdelaiDet`, it is also necessary to clone my custom branch (I'm considering to merge the `quantization` branch in my repo to the official repo if it is possible).

```
cd /workspace/git/
git clone https://github.com/blueardour/uofa-AdelaiDet AdelaiDet
# notice to change to the quantization branch
cd AdelaiDet
git checkout quantization
```

The custom project [custom detectron2](https://github.com/blueardour/detectron2) and [custom AdelaiDet](https://github.com/blueardour/uofa-AdelaiDet) will upgrade regularly from origin repo.

As the orignal project, `custom AdelaiDet` depends on `custom detectron2`.  Install those two projects based on the original install instructions.

2. install dependent package according to [classification.md](./classification.md)

3. make sure the symbol link is correct.
```
cd /workspace/git/detectron2
ls -l third_party
# the third_party/quantization should point to /workspace/git/model-quantization/models
```

Currently I link the dependentant with symbol link. As these projects will update separatedly, submodule with version management is considered when all scripts being already.

## Dataset

refer [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)

## What is modified in the detectron2 project

The main modification of the project to add quantization support lay in the followsing files.  Use `vimdiff` to check the difference.

```
modified:   detectron2/checkpoint/detection_checkpoint.py
modified:   detectron2/config/defaults.py
modified:   detectron2/engine/defaults.py
modified:   detectron2/engine/train_loop.py
modified:   detectron2/layers/wrappers.py
modified:   detectron2/modeling/backbone/fpn.py
modified:   detectron2/modeling/meta_arch/build.py
new file:   third_party/convert_to_quantization.py
new file:   third_party/quantization
```

## Training and Test

Training and testing methods follow original projects ( [detectron2](https://github.com/facebookresearch/detectron2) or [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) ). Just adapt the quantization need by modifying the configration file.

Example configurations for quantization are provided in `detectron2/config` and `AdelaiDet/config` . In `detectron2` and `aim-uofa/AdelaiDet` project, most of the options are managed by the `yaml` config file. Thus, the `detectron2/config/default.py` is modified to add the quantization related options. They have the same meaning with the ones in classification task. Refer option introduction in [classification.md](./classification.md#Training-script-options)

## Speical Guide for quantization

The overall flow of the quantization on detection/ segmentation tasks are as follows, some of them can be omit if pretrained model alreay exist.

- Train full precision backbone network on Imagenet

  Refer the resulted model as `backbone_full.pt`

- Finetune the low bit model (backbone network)

  Refer [classification.md](./classification.md) for fintuning with `backbone_full.pt` as initilization.
  
  Refer the resulted model as `backbone_low.pt`
  
- Export `backbone_full.pt` and `backbone_low.pt` detectron2 project format. 

  To import customed pretrained model / pytorch resnet paramters to this project, refer the `renaming function` provide in `tools.py` demonstrated in [tools.md](./tools.md)

- Train in full precision of the detection /segmentation task with formatted `backbone_full.pt` as initilization.
  
  Refer the resulted model as `overall_full.pt`
 
 - Finetune the detection /segmentation model with double initilization for quantization.  
 
   We provide `WEIGHT_EXTRA` option to load an extra pretrain model. When quantization, provide the `overall_full.pt` as extra initilization. Also, override some of the initilization with another pretrianed model - the formatted `backbone_low.pt`.

## Special Notice on the Structure of Quantizaiton

The performance of quantization network is approved to be possible improved with the following tricks.

- Employ normalization (such as BatchNorm) and no-linearity (such as ReLU) to the FPN module.

- Empoly normalization (such as GroupNorm or BatchNorm) to the tower in the Head module. (No-share BatchNorm is demonstrate the superior performance)

## License and contribution 

See [README.md](../README.md)
