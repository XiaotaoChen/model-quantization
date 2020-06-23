# Quantization object detection/segmentation tasks

## Install

1. download the [custom detectron2 project](https://github.com/blueardour/detectron2). See what is modified below.

```
cd /workspace/git/
git clone https://github.com/blueardour/detectron2
```

Install it based on the original [install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

2. install dependent package according to [classification.md](./classification.md)

## Dataset

refer [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)

### What is modified

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
