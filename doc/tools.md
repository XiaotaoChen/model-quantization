
# A independent script for debug / model converting

Selected functions which is commonly used.

## Model import and paramter renaming.

Some times models are trained by the old repo. When importing the pretrained model to new repo, some variable /paramters name might be changed.

Following the commands for looking up items and converting.

1. looking up item in the model file.

```
python tools.py --keyword verbose --verbose_list all --pretrained [pretrained.pt]
```

2. renaming parameter


```
python tools.py --keyword update[,raw]  --mf [mapping from file] --mt [mapping to file] --old [old model.pt] --new [save model to.pt]
```

