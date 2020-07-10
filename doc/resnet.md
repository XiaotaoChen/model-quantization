
# Resnet Architecture

To use the resnet architecture, one can set the `arch` option to either `resnet18/34/50/101` or `pytorch-resnet18/34/50/101`. The former is realized in `model/resnet_.py` and the later is implemented in `model/resnet.py`.  The difference between the choice is that the later one is from the pytorch offical and the former is more flexible. Users can customize the Resnet architecture variants with different combination of options for `resnet18/34/50/101`. The combination is possible by setting segments in `keyword`.

For example, by add `origin,cbsa,fix_pooling,singleconv,fix` in the keyword, we obtain the pytorch official architecture (the same with the `pytorch-resnet18/34/50/101`). Fine-grain control is supported for the stem and body.

## Stem

![stem](./1.jpg)

## Body

- Default buildup

![stem](./2.jpg)

- Lossless downsample network

![stem](./3.jpg)

- Prone: Point-wise and Reshape Only Network

![stem](./4.jpg)

- Specific Activation and Normalization

![stem](./5.jpg)

