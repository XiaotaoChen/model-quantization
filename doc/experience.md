
## Training tricks

- Learning rate

- Activation Function

- Initilization

  For quantization in all kinds of tasks, it is advised to first train the full precision model and then quantization by fintuning with the full precision model as the initilization.

  This strategy shows little improvement in the BNN training for image classification task, however, poses considerable benefit for higher bit quantization. It is specially important for detection and segmentation tasks.

  ***More specifically, for detection and segmentation tasks, double initilization is highly recommended.***  See [detection.md](./detection.md#Speical-Guide-for-quantization)
  
  Another trick about the Initilization is the `Initilization according statistic on small amount of data`. It means, before the finetuning procedure, we first take samples in the dataset and initilize the custom varibales (for example, the `clip_val` in Dorefa-net based methods or the `basis` in LQ-net) based these samples. We provide the support by add the `stable`, `warmup` and `custom-update` options.


## Speed on real platform

I developed acceleration code of low bit quantization and evaluted the quantized model on real platforms.
The profiling result on embedded side devices (current tested on Huawei Hisilicon and Qualcomm Snapdragon) and PC-side GPU cards (Nvidia)
implies than the real efficiency is far beyond the theory analysis.

For example, the `FLOPS` statistics in many publiction to demonstrate their computational saving is not the real case.
The commonly used claim in lots of paper that 'We improve the quantization accuracy by a large margin 
by employing `high precision branch` or `elaborately designed structure` which has only a neglectable FLOPS consumption.'
seems not feasible on real platforms. Benchmark result indicates the necessity of implementation driven exploring for quantization.

