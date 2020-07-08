

## Instance Segmentation

For training and inference instructions, refer [detection.md](./detection.md).

Dataset | Task Method | Quantization method | Model | A/W | Reported | BBox AP / Seg AP  | Comment 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
COCO | Blenmask | LSQ | Torch-18 | 32/32 | - | 32.3/29.1 | 1x,550-R-18-Full-BN
COCO | Blenmask | LSQ | Torch-18 | 2/2 | - | 27.5/25.2 | 1x,550-R-18-Full-BN, Quantize-Detector, doube-init
COCO | Blenmask | LSQ | Torch-18 | 2/2 | - | 25.3/23.0 | 1x,550-R-18-Full-BN, Quantize-All, doube-init
