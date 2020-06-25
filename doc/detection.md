# Quantization object detection/segmentation tasks

## Dashboard

The quantization performance demoonstration.

## Install

1. download the [custom detectron2](https://github.com/blueardour/detectron2) project. See what is modified below.

```
cd /workspace/git/
git clone https://github.com/blueardour/detectron2
# checkout the quantization branch
cd detectron2
git checkout quantization
```
Facebook detectron2 has not support for some works such as `FCOS`. Try the [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for more task support. Note, for the `aim-uofa/AdelaiDet`, it is also necessary to clone my custom branch (I'm considering to merge the `quantization` branch in my repo to the official repo if it is possible).

```
cd /workspace/git/
git clone https://github.com/blueardour/uofa-AdelaiDet AdelaiDet
# notice to change to the quantization branch
cd AdelaiDet
git checkout quantization
```

The custom project [custom detectron2](https://github.com/blueardour/detectron2) and [custom AdelaiDet](https://github.com/blueardour/uofa-AdelaiDet) will upgrade regularly from origin repo.

As the orignal project, `custom AdelaiDet` depends on `custom detectron2`.  Install those two projects based on the original install instuctions.

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

Example configurations for quantization are provided in `AdelaiDet/config`

## License and contribution 

See [README.md](../README.md)
