
# Deep learning model Compression

This project collects and trys to reproduce the result of the model quantization algorithms in the publication.
If you find it useful, please cite the following work / the specific work you are employing.


## How to use

1. create necessary folder and prepare the datasets. Advised structure:

```
# dataset
mkdir -p /data/imagenet/{train, val}
# download imagnet in /data/imagenet/

# log and weight:
# download the pretrained model and put into the /data/pretrained/pytorch/model-quantization/weights/ folder
mkdir -p /data/pretrained/pytorch/model-quantization/{exp, weights}

#code: download and move the code to /workspace/git/model-quantization
cd /workspace/git/
git clone https://github.com/blueardour/model-quantization
git clone https://github.com/blueardour/pytorch-utils
cd model-quantization
ln -s ../pytorch-utils utils
ln -s /data/pretrained/pytorch/model-quantization/exp .
ln -s /data/pretrained/pytorch/model-quantization/weights .
```

2. basic prerequisite packages are listed in requirement.txt (python 3+).

Additional packages include the [Nvidia Dali](https://github.com/NVIDIA/DALI) and [Nvidia Mix precision training package](https://github.com/NVIDIA/apex)

3. to train or test (modify the train.sh according to your own evironment if not following above folder structure):

```
bash train.sh config.xxxx
```

`config.xxxx` is the configuration file.
Default naming pattern is `config` + `method` + `phase` + `dataset` + `precision` + `network`.

where `phase` contains `eval` for evaluation; `train-stratch` for training without pretrained model; `finetuning` means training with pretrained model as the initilization.

`dataset` can be chosen from `imagenet` for the imagenet dataset; `dali` for the imagenet dataset wrappered in nvidia dali for fast preprocessing; `cifar10` for CIFAR10 dataset; `cifar100` for CIFAR100 dataset and `fake` for fake images for testing.

For example, `config.dorefa.eval.imagenet.fp.resnet18` inidcates to evaluate the ResNet-18 network on imagenet dataset with full precision.

Run `bash train.sh config.dorefa.eval.imagenet.fp.resnet18` to test the PreBN ResNet-18 on imagenet. Expected accuracy: Top-1(70.1) and Top-5(89.3).

## Algorithms

The project consists of 

1. LQnet:

Method | Model | A/W | Paper Reported | My Top-1  | Comment 
--- |:---:|:---:|:---:|:---:|:---:
LQ-net | ResNet-18 | 2/2 | 64.9 | 64.9 | PreBN,bacs 
LQ-net | ResNet-18 | 2/2 | - | 65.9 | PreBN,bacs,wt-qg=8
--- |:---:|:---:|:---:|:---:|:---:

To test:

```
bash train.sh config.lq-net.eval.dali.2bit.resnet18
```

```
bash train.sh config.lq-net.eval.dali.2bit.resnet18-fg8
```


Please cite if you use this method

```BibTeX
@inproceedings{zhang2018lq,
  title={LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks},
    author={Zhang, Dongqing and Yang, Jiaolong and Ye, Dongqiangzi and Hua, Gang},
      booktitle=ECCV,
        year={2018}
}
```


2. Dorefa-net

Please cite if you use this method

```BibTeX
```

3. PACT

Please cite if you use this method

```BibTeX
```


4. LSQ/TET

Please cite if you use this method

```BibTeX
```


5. Group-net

Please cite if you use this method

```BibTeX
```


6. Xnor-net

Please cite if you use this method

```BibTeX
```


7. Bi-Real net

Please cite if you use this method

```BibTeX
```

8. FATNN

Please cite if you use this method

```BibTeX
```

## to be contiune



