
dataset='cifar100'
root=$FASTDIR/data/cifar

model='resnet20'
options="$options --width_alpha 0.25"

train_batch=256
val_batch=50

case='cifar100--lsq-finetune-2bit-pytorch-TypeD-wd1e-4-wt_qg1_var-mean-sgd_0'
keyword='cifar100,origin,cbsa,fix_pooling,singleconv,fix,ReShapeResolution,real_skip,dorefa,lsq'

pretrained='cifar100-EfficientQuant-stratch-fp-pytorch-TypeD-sgd_0-model_best.pth.tar'
options="$options --pretrained $pretrained"

 options="$options --tensorboard"
 options="$options --verbose"
#options="$options -j2"
#options="$options -e"
#options="$options -r"
#options="$options --fp16 --opt_level O1"
 options="$options --wd 1e-4"
 options="$options --decay_small"
#options="$options --custom_lr_list alpha --custom_lr 1e-3"
 options="$options --order cba"

 epochs=200
  options="$options --fm_bit 2 --fm_enable"
 options="$options --wt_bit 2 --wt_enable"
 options="$options --fm_quant_group 1"
 options="$options --wt_quant_group 1"
 options="$options --wt_adaptive var-mean"

# SGD
 options="$options --lr 1e-2   --lr_policy custom_step --lr_decay 0.2 --lr_custom_step 60,120,160 --nesterov"

#options="$options --wt_quant_group 1"
#options="$options --wt_adaptive var-mean"

