
# A independent script for debug / model converting

This page presents selected functions which is commonly used.

## Model import and paramter renaming.

Some times models are trained by the old repo. When importing the pretrained model to new repo, some variable /paramters name might be changed.

Following the commands for looking up items and converting (Remove the brackets when testing, brackets only indicates the enclosed one can be replaced with other string).

1. looking up item in the model file.

```
# cd weights/pytorch-resnet50/
# download pretrained model by
# wget https://download.pytorch.org/models/resnet50-19c8e357.pth
python tools.py --keyword verbose --verbose_list all --old [weights/pytorch-resnet50/resnet50-19c8e357.pth]
```

2. renaming parameter

Export pytorch official resnet model to Detectron2 format as initilization model. Edit your own `mapping_from.txt` and `mapping_from.txt` file based on the naming space (which can be verbosed by above command)
```
python tools.py --keyword update[,raw]  --mf [weights/det-resnet50/mapping_from.txt] --mt [weights/det-resnet50/mapping_to.txt] --old [weights/pytorch-resnet50/resnet50-19c8e357.pth] --new [weights/det-resnet50/official-r50.pth]
```

Add `raw` in the keyword to generate the model file with/without `state_dict` segment.
