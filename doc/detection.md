# Quantization for various computer vision tasks

The framework is able to provide quantization support for all kinds of tasks that the `Detectron2` and `AdelaiDet` projects integrate. Mix precision training is also available as a benefit.


## Install

1. install dependent packages according to [classification.md](./classification.md)

***Note a known issue for the FP16 training: Training with FP16 and SyncBN on multi-GPU seems to cause NAN loss for current projects. Use normal BN instead***

2. download the [custom detectron2](https://github.com/blueardour/detectron2) project. See what is modified below.

```
cd /workspace/git/
git clone https://github.com/blueardour/detectron2
# checkout the quantization branch
cd detectron2
git checkout quantization

# install 
pip install -e .

### other install options
## (add --user if you don't have permission)
#
## Or if you are on macOS
#CC=clang CXX=clang++ python -m pip install ......


# link classification pretrained weight
ln -s ../model-quantization/weights .
```
Facebook detectron2 does not support some works such as `FCOS` and `Blendmask`. Try the [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for more tasks. Note, for the `aim-uofa/AdelaiDet`, it is also necessary to clone my custom branch (I'm considering to merge the `quantization` branch in my repo to the official repo if it is possible).

```
cd /workspace/git/
git clone https://github.com/blueardour/uofa-AdelaiDet AdelaiDet
# notice to change to the quantization branch
cd AdelaiDet
git checkout quantization

# install
python setup.py build develop

# link classification pretrained weight
ln -s ../model-quantization/weights .
```

The custom project [custom detectron2](https://github.com/blueardour/detectron2) and [custom AdelaiDet](https://github.com/blueardour/uofa-AdelaiDet) will upgrade regularly from the origin repo.

Similar with the orignal project, `custom AdelaiDet` depends on `custom detectron2`.  Install the two projects based on the original install instructions.

3. make sure the symbol link is correct.
```
cd /workspace/git/detectron2
ls -l third_party
# the third_party/quantization should point to /workspace/git/model-quantization/models
```

## Dataset

refer detectron2 datasets: [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md)

and specific datasets from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)

## Pretrained model and quanzation Results

- [Detection](./result_det.md)

- [Segmentation](./result_seg.md)

We provide pretrained models gradually in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)

## What is modified in the detectron2 project

The `model-quantization` project is used as a plugin to other projects to provide the quantization support. We modify the following files to integrate the `model-quantization` project into the `detectron2` / `AdelaiDet` projects. Use `vimdiff` to check the difference. The `model-quantization` project is potential to be equipped into other projects in a similar way.

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

Highly recommend to check the `detectron2/engine/defaults.py` to see which options are added for the low-bit quantization.

```
git difftool quantization master detectron2/config/defaults.py
```



## Training and Test

  Training and testing methods follow original projects ( [detectron2](https://github.com/facebookresearch/detectron2) or [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) ).  To obtain the quantization version of given models, please modify corresponding configuration files by setting quantization related options.

  Example configurations for quantization are provided in `detectron2/config` and `AdelaiDet/config`. In `detectron2` and `aim-uofa/AdelaiDet` projects, most of the options are managed by the `yaml` config file. Thus, the `detectron2/config/default.py` is modified to add the quantization related options. They have the same meaning with the ones in classification task. Refer option introduction in [classification.md](./classification.md#Training-script-options)

  If you want to test the low-bit quantization model only, just download the pretrained model and run the test. If training is required, see below [examples](./detection.md#Examples) for demonstration.

## Speical Guide for quantization

  The overall flow of the quantization on detection/ segmentation / text spotting tasks are as follows, some of them can be omit if the pretrained model already exists.

- Train full precision backbone network on Imagenet

  Refer the saved model as `backbone_full.pt`

- Finetune the low-bit model (backbone network)

  Refer [classification.md](./classification.md) for finetuning with `backbone_full.pt` as initialization.
  
  Refer the saved model as `backbone_low.pt`
  
- Import `backbone_full.pt` and `backbone_low.pt` into detectron2 project format. 

  To import the pretrained models in correct format, refer the `renaming function` provided in `tools.py` demonstrated in [tools.md](./tools.md) and also the [examples](./detection.md#Examples).

- Train the full precision model with formatted `backbone_full.pt` as initialization.
  
  Refer the saved model as `overall_full.pt`
 
 - Finetune low-bit model with double pass initialization (`overall_full.pt` and `backbone_low.pt`) or single pass initialization (`overall_full.pt`).

## Examples

### Detection

- Resnet18-FCOS Quantization by LSQ into 2-bit model

1. Pretrain the full precision and 2-bit backbone in the [`model-quantization`](https://github.com/blueardour/model-quantization) project. We provide ResNet-18/50 pretrained models in the download link. Prepare your own model if other backbones are required. 
   
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
  
  The `mf.txt` and `mt.txt` files for the Resnet18 are uploaded in the `model-quantization` project as an example. The files for Resnet50 are also provided. Refer [tools.md](./tools.md) for more instructions.

3. train full precision FCOS-R18-1x

  Check the configuration file `configs/FCOS-Detection/R_18_1x-Full-SyncBN.yaml`
  
  ```
  cd /workspace/git/AdelaiDet
  # add other options, such as the GPU number as needed
  python tools/train_net.py --config-file configs/FCOS-Detection/R_18_1x-Full-SyncBN.yaml
  ```
  
  ***Check the parameters on the backbone are re-loaded correctly***

  This step would obtain the pretrained model in `output/fcos/R_18_1x-Full-SyncBN/model_final.pth`

4. fintune to get quantization model

  Check the configuration file `configs/FCOS-Detection/R_18_1x-Full-SyncBN-lsq-2bit.yaml`
  
  ```
  cd /workspace/git/AdelaiDet
  # add other options, such as the GPU number as needed
  python tools/train_net.py --config configs/FCOS-Detection/R_18_1x-Full-SyncBN-lsq-2bit.yaml
  ```
  
  ***Check the parameters in double initialization are re-loaded correctly***
  
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
