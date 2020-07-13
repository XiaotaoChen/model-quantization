
dataset='imagenet'
root=$FASTDIR/data/imagenet

model='resnet18'
#options="$options --width_alpha 0.25"

train_batch=256
val_batch=50

case='imagenet-ldn-stratch-fp-pytorch-TypeD-sgd_0'
keyword='imagenet,origin,cbsa,fix_pooling,singleconv,fix,ReShapeResolution,real_skip'

pretrained='None'
options="$options --pretrained $pretrained"

 options="$options --tensorboard"
 options="$options --verbose"
#options="$options -j2"
#options="$options -e"
 options="$options -r"
#options="$options --fp16 --opt_level O1"
 options="$options --wd 1e-4"
 options="$options --decay_small"
#options="$options --custom_lr_list alpha --custom_lr 1e-3"
 options="$options --order cba"

 epochs=90
# SGD
 options="$options --lr 1e-1 --lr_policy custom_step --lr_decay 0.1 --lr_custom_step 30,60,80 --nesterov"
#options="$options --lr 1e-1 --lr_policy sgdr --lr_custom_step 90 --eta_min 1e-6 --nesterov"
#options="$options --lr 1e-1 --lr_policy sgdr --lr_custom_step 6  --eta_min 1e-6 --nesterov"


