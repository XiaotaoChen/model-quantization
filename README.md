
# Deep learning model Compression

This project collects and trys to reproduce the result of the model quantization algorithms in the publication.
If you find it useful, please cite the following work / the specific work you are employing.


## How to
To begin:

1. create necessary folder and prepare the datasets. Advised structure:

#dataset: /data/imagenet/{train, val}
   
#log and weight: /data/pretrained/pytorch/model-quantization/{exp, weights}
#download the pretrained model and put into the /data/pretrained/pytorch/model-quantization/weights/ folder

#code: download and move the code to /workspace/git/model-quantization

```
cd /workspace/git/
git clone https://github.com/blueardour/model-quantization
git clone https://github.com/blueardour/pytorch-utils
cd model-quantization
ln -s ../pytorch-utils utils
ln -s /data/pretrained/pytorch/model-quantization/exp .
ln -s /data/pretrained/pytorch/model-quantization/weights .
```

2. install prerequisite packages listed in requirement.txt in python 3.

3. to train or test (modify the train.sh according to your own evironment if not following above folder structure):

```
bash train.sh config.xxxx
```

Examples of config.xxxx is illustrated below.

## Algorithms

The project consists of 

1. LQnet:

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



