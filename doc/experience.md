
## Training tricks

- Learning rate

- Activation Function

- Initilization


## Speed on real platform

I developed acceleration code of low bit quantization and evaluted the quantized model on real platforms.
The profiling result on embedded side devices (current tested on Huawei Hisilicon and Qualcomm Snapdragon) and PC-side GPU cards (Nvidia)
implies than the real efficiency is far beyond the theory analysis.

For example, the `FLOPS` statistics in many publiction to demonstrate their computational saving is not the real case.
The commonly used claim in lots of paper that 'We improve the quantization accuracy by a large margin 
by employing `high precision branch` or `elaborately designed structure` which has only a neglectable FLOPS consumption.'
seems not feasible on real platforms. Benchmark result indicates the necessity of implementation driven exploring for quantization.

