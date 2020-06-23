# Quantization object detection/segmentation tasks

## Install

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
