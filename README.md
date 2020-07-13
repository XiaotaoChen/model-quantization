
# QTool: A low-bit quantization toolbox for deep neural networks in computer vision

This project provides aboundant choices of quantization strategies (such as the quantization algoirthms, training schedules and empirical tricks) for quantizing the image classification neural networks into low-bit counterparts. Associated projects demonstrate that this project can also benefit other computer vision tasks, such as object detection, segmentation and text parsing. Pretrained models are provided to show high standard of the code on achiving appealing quantization performance. 

## Supported Task

- [Classification](./doc/classification.md): [Performance](./doc/result_cls.md)
- [Detection](./doc/detection.md): [Performance](./doc/result_det.md)
- [Segmentation](./doc/detection.md): [Performance](./doc/result_seg.md)
- [Text parsing](./doc/detection.md): [Performance](./doc/result_text.md)

## Update History

- Text parsing (preparing)
- 2020.07.08 Instance Segmentation
- 2020.07.08 Object Detection
- 2020.06.23 Add classification quantization

## Citation

Please cite the following work if you find the project helpful.

```
@misc{chen2020qtool,
author = {Peng Chen, Bohan Zhuang, Jing Liu and Chunlei Liu},
title = {{QTool: A low-bit quantization toolbox for deep neural networks computer vision}},
year = {2020},
howpublished = {\url{https://github.com/blueardour/model-quantization}},
note = {Accessed: [Insert date here]}
}
```


Also cite the corresponding publications when you choose [dedicated algorithms](./doc/reference.md).

## Contribute

To contribute, PR is appreciated and suggestions are welcome to discuss. Private contact is available at blueardour@gmail.com

## License

For academic use, this project is licensed under the 2-clause BSD License. See LICENSE file.

