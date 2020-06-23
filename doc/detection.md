# Quantization object detection/segmentation tasks

## Install

1. download the [custom detectron2](https://github.com/blueardour/detectron2) project. See what is modified below.

```
cd /workspace/git/
git clone https://github.com/blueardour/detectron2
```
Facebook detectron2 has not support for some works such as `FCOS`. Try the [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for more task support. Note, for the `aim-uofa/AdelaiDet`, it is also necessrary to clone my custom branch.

```
cd /workspace/git/
git clone https://github.com/blueardour/uofa-AdelaiDet
git checkout quantization
```

The custom project [custom detectron2](https://github.com/blueardour/detectron2) and [custom AdelaiDet](https://github.com/blueardour/uofa-AdelaiDet) will upgrade regularly from origin repo.

As the orignal project, `custom AdelaiDet` depends on `custom detectron2`.  Install those two projects based on the original install instuctions.

2. install dependent package according to [classification.md](./classification.md)

## Dataset

refer [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)

## What is modified

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

Training and testing follow original projects ( [detectron20(https://github.com/facebookresearch/detectron2) or [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) )
