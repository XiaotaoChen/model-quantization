# Quantization various computer vision tasks

The framework is able to provide quantization support for all kinds of tasks that the `Detectron2` and `AdelaiDet` projects integrate.

## Dashboard

Here lists selected experiment result. The performance is potentially being better if more effort is paid on tuning. See [experience.md](experience.md) to communicate training skills with me.

### Detection
Dataset | Task Method | Quantization method | Model | A/W | Reported | AP  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:

### Instance Segmentation
Dataset | Task Method | Quantization method | Model | A/W | Reported | AP  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:

### Text spotting
Dataset | Task Method | Quantization method | Model | A/W | Reported | AP  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:

***The following table is based on an old repo, Refer other tables for latest result***

### Detection
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

1. install dependent package according to [classification.md](./classification.md)

2. download the [custom detectron2](https://github.com/blueardour/detectron2) project. See what is modified below.

```
cd /workspace/git/
git clone https://github.com/blueardour/detectron2
# checkout the quantization branch
cd detectron2
git checkout quantization

# install 
pip install -e .
```
Facebook detectron2 has not support for some works such as `FCOS` and `Blendmask`. Try the [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for more task support. Note, for the `aim-uofa/AdelaiDet`, it is also necessary to clone my custom branch (I'm considering to merge the `quantization` branch in my repo to the official repo if it is possible).

```
cd /workspace/git/
git clone https://github.com/blueardour/uofa-AdelaiDet AdelaiDet
# notice to change to the quantization branch
cd AdelaiDet
git checkout quantization

# install
python setup.py build develop

```

The custom project [custom detectron2](https://github.com/blueardour/detectron2) and [custom AdelaiDet](https://github.com/blueardour/uofa-AdelaiDet) will upgrade regularly from origin repo.

Similar with the orignal project, `custom AdelaiDet` depends on `custom detectron2`.  Install those two projects based on the original install instructions.

3. make sure the symbol link is correct.
```
cd /workspace/git/detectron2
ls -l third_party
# the third_party/quantization should point to /workspace/git/model-quantization/models
```

Currently I link the dependentant with symbol link. As these projects will update separatedly, submodule with version management is considered when all scripts being already.

## Dataset

refer detectron2 datasets: [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)

and or specific datasets from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)

## What is modified in the detectron2 project

The main modification of the project to add quantization support lays in the followsing files.  Use `vimdiff` to check the difference.

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

See above [examples](./detection.md#Examples) for demonstration.

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

- Quantization is employed on all convolution layer wrapper in `detectron2/layer/wrapper.py`, namely the `Conv2D` module. For layers natively call `nn.conv2d` will keep in full precision.

- We provide an option `quantization.scope` to flexible choose the layers/blocks which are scheduled to be quantized.


## Pretrained  model

We provide pretrained models gradually in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)

## Examples

### Detection

- Resnet18-FCOS Quantization by LSQ into 2-bit model

1. imagenet full precision and 2-bit LSQ quantization model draw from classification project (download pretrained model from [classification.md](./classification.md))
   Prepare your own model if other configuraton is required
   
   full precision model: `weights/pytorch-resnet18/resnet18_w32a32.pth`
   
   2-bit LSQ model: `weights/pytorch-resnet18/lsq_best_model_a2w2.pth`
   
2. import model from classification project to detection project.

```
cd /workspace/git/model-quantization
# prepare the weights/det-resnet18/mf.txt and weights/det-resnet18/mt.txt
# the two files are created manually with the parameter renaming
python tools.py --keyword update,raw --mf weights/det-resnet18/mf.txt --mt weights/det-resnet18/mt.txt --old weights/pytorch-resnet18/resnet18_w32a32.pth --new weights/det-resnet18/resnet18_w32a32.pth

python tools.py --keyword update,raw --mf weights/det-resnet18/mf.txt --mt weights/det-resnet18/mt.txt --old weights/pytorch-resnet18/lsq_best_model_a2w2.pth --new weights/det-resnet18/lsq_best_model_a2w2.pth
```

3. train full precision FCOS-R18-1x

Check the configuration file `configs/FCOS-Detection/R_18_1x-Full-SyncBN.yaml`

```
cd /workspace/git/AdelaiDet
# add other options, such as the GPU number as needed
python tools/train_net.py --config-file configs/FCOS-Detection/R_18_1x-Full-SyncBN.yaml
```

***Check the parameters on the backbone are re-loaded correctly****

This step would obtain the pretrained model in `output/fcos/R_18_1x-Full-SyncBN/model_final.pth`

4. fintune to get quantization model

Check the configuration file `configs/FCOS-Detection/R_18_1x-Full-SyncBN-lsq-2bit.yaml`

```
cd /workspace/git/AdelaiDet
# add other options, such as the GPU number as needed
python tools/train_net.py --config configs/FCOS-Detection/R_18_1x-Full-SyncBN-lsq-2bit.yaml
```

***Check the parameters in double initialization are re-loaded correctly****

Compare the accuracy with the one in step 3.

### Segmentation

- Resnet18-Blendmask Quantization by LSQ into 2-bit model

  Similar with the detection flow, but with different configuration file
  ```
  cd /workspace/git/AdelaiDet
  # add other options, such as the GPU number as needed
  # full precision pretrain
  python tools/train_net.py --config configs/BlendMask/550_R_18_1x-full_syncbn.yaml
  # finetune
  python tools/train_net.py --config configs/BlendMask/550_R_18_1x-full_syncbn-lsq-2bit.yaml
  ```


## License and contribution 

See [README.md](../README.md)
