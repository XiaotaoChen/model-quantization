import os, sys, glob, argparse
import logging
from collections import OrderedDict
import torch

import utils
import models
import main as entry

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def export_onnx(args):
    model_name = args.model
    if model_name in models.model_zoo:
        model, args = models.get_model(args)
    else:
        print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
        return

    if utils.check_file(args.pretrained):
        print("load pretrained from %s" % args.pretrained)
        checkpoint = torch.load(args.pretrained)
        print("load pretrained ==> last epoch: %d" % checkpoint.get('epoch', 0))
        print("load pretrained ==> last best_acc: %f" % checkpoint.get('best_acc', 0))
        print("load pretrained ==> last learning_rate: %f" % checkpoint.get('learning_rate', 0))
        try:
            utils.load_state_dict(model, checkpoint.get('state_dict', None))
        except RuntimeError:
            print("Loading pretrained model failed")
    else:
        print("no pretrained file exists({}), init model with default initlizer".
            format(args.pretrained))

    onnx_model = torch.nn.Sequential(OrderedDict([
        ('network', model),
        ('softmax', torch.nn.Softmax()),
    ]))

    onnx_path = "onnx/" + model_name
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)
    onnx_save = onnx_path + "/" + model_name + '.onnx'

    input_names = ["input"]
    dummy_input = torch.zeros((1, 3, args.input_size, args.input_size))
    output_names = ['prob']
    torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_save,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=7,
            keep_initializers_as_inputs=True
            )

def histogram_weight(args):
    from tensorboardX import SummaryWriter
    utils.check_folder('debug')
    tensorboard = SummaryWriter('debug')
    if torch.cuda.is_available():
        disk_checkpoint = torch.load(args.pretrained)
    else:  # force cpu mode
        disk_checkpoint = torch.load(args.pretrained, map_location='cpu')
    checkpoint = disk_checkpoint['state_dict']
    for name, value in checkpoint.items():
        if ('quant_activation' in name or 'quant_weight' in name) and name.split('.')[-1] in args.verbose_list:
            print(name, value.shape, value.requires_grad)
            print(value.data)

def finetune_sgdr(args):
    args.epochs = 80
    args.weight_decay = 0
    args.lr = 5e-4
    args.lr_policy = 'sgdr'
    args.lr_custom_step = '5,15,35'
    return args

def finetune_fix_step(args):
    args.epochs = 80
    args.weight_decay = 0
    args.lr = 5e-4
    args.lr_policy = 'custom_step'
    args.lr_custom_step = '20,40,60'
    args.lr_decay = 0.5
    return args

def relax_sgdr(args):
    #args.weight_decay = 0
    args.epochs = 120
    args.eta_min = 1e-6
    args.lr = 2e-2
    args.lr_policy = 'sgdr'
    args.lr_custom_step = '6,18,42'
    return args

def relax_fix_step(args):
    #args.weight_decay = 0
    args.epochs = 120
    args.lr = 2e-2
    args.lr_policy = 'custom_step'
    args.lr_custom_step = '30,60,90'
    args.lr_decay = 0.2
    return args

def load_finetune_default():
    args = entry.get_parameter()
    args.dataset = 'cifar100'
    args.root = '/workspace/data/cifar'
    args.model = 'resnet18'
    args.base = 1

    args.batch_size = 256
    args.val_batch_size = 100
    args.save_freq = -1
    args.nesterov = True
    args.eta_min = 1e-6
    args.resume = False
    args.tensorboard = True
    args.optimizer = 'ADAM'

    args.fm_separator = 0.5
    args.wt_separator = 0.5
    args.wt_range = 1.2
    args.fm_range = 1.2
    args.fm_ratio = 1.6
    args.wt_ratio = 1.6
    args.fm_enable = True
    args.wt_enable = True

    args.keyword = 'cifar100,ternary,casb,pretrain'
    args = finetune_fix_step(args)
    #args = finetune_sgdr(args)
    return args

def load_relax_default():
    args = entry.get_parameter()
    args.dataset = 'cifar100'
    args.root = '/workspace/data/cifar'
    args.model = 'resnet18'
    args.base = 1

    args.batch_size = 256
    args.val_batch_size = 100
    args.save_freq = -1
    args.nesterov = True
    args.resume = False
    args.tensorboard = True
    #args.optimizer = 'ADAM'

    args.fm_separator = 0.5
    args.wt_separator = 0.5
    args.wt_range = 1.2
    args.fm_range = 1.2
    args.fm_ratio = 1.6
    args.wt_ratio = 1.6
    args.fm_enable = True
    args.wt_enable = True

    args.keyword = 'cifar100,ternary,casb,pretrain'
    args = relax_fix_step(args)
    #args = finetune_sgdr(args)
    return args

def batch_finetune_cifar100():
    index = 1
    last_case_index = index
    args = load_relax_default()
    args.bitA = 2
    args.pretrained = 'cifar100-pretrain-ter_relax_v25-casb-model_best.pth.tar'
    args.case = "cifar100-ter_debug_finetune_%d" % index
    entry.main(args)

    index = 2
    last_case_index = index
    args = load_relax_default()
    args.bitA = 2
    args.wt_enable = False
    args.pretrained = 'cifar100-ter_debug_finetune_%d-model_best.pth.tar' % last_case_index
    args.case = "cifar100-ter_debug_finetune_%d" % index
    entry.main(args)

def batch_debug():
    batch_finetune_cifar100()

def re_anchor():
    args = load_relax_default()
    args.dataset = 'cifar10'
    args.model = 'vgg_small'
    args.batch_size = 128
    args.tensorboard = False
    args.keyword = 'cifar10,relax,group,abc'
    args.case = 'cifar10-group-relax-ter-debug_5'
    args.pretrained='cifar10-group-relax-ter-debug_5-model_best.pth.tar'
    args.epochs = 0
    args.fm_bit = 1.5 
    args.wt_bit = 1.5 
    args.fm_half_range=0
    args.re_anchor = True
    args.verbose = True
    entry.main(args)

def get_parameter():
    parser = utils.get_parser()
    parser.add_argument('--base', default=1, type=int)
    parser.add_argument('--with-softgates', action='store_true', default=False)
    parser.add_argument('--relax_separator', default=0, type=float)
    parser.add_argument('--relax_ratio', default=1, type=float)
    parser.add_argument('--relax_scale', action='store_true', default=False)
    parser.add_argument('--old', type=str, default='')
    parser.add_argument('--new', type=str, default='')
    parser.add_argument('--mapping_from', '--mf', type=str, default='')
    parser.add_argument('--mapping_to', '--mt', type=str, default='')
    parser.add_argument('--verbose_list', default='ratio,sep', type=str)
    args = parser.parse_args()
    if isinstance(args.verbose_list, str):
        args.verbose_list = [x.strip() for x in args.verbose_list.split(',')]
    if isinstance(args.keyword, str):
        args.keyword = [x.strip() for x in args.keyword.split(',')]
    return args

def main():
    args = get_parameter()
    args.weights_dir = os.path.join(args.weights_dir, args.model)
    utils.check_folder(args.weights_dir)

    utils.setup_logging(os.path.join(args.log_dir, 'tools.txt'), resume=True)
    #args.old = os.path.join(args.weights_dir, args.old)
    #args.new = os.path.join(args.weights_dir, args.new)

    config = dict()
    for i in args.keyword:
        config[i] = True

    if 'export_onnx' in config.keys():
        export_onnx(args)

    if 're-anchor' in config.keys():
        re_anchor()

    if 'debug' in config.keys():
        batch_debug()

    if 'batch-finetune-cifar100' in config.keys():
        batch_finetune_cifar100()

    if 'batch-relax-cifar100' in config.keys():
        batch_relax_cifar100()

    if 'verbose' in config.keys():
        if torch.cuda.is_available():
            disk_checkpoint = torch.load(args.pretrained)
        else:  # force cpu mode
            disk_checkpoint = torch.load(args.pretrained, map_location='cpu')
        checkpoint = getattr(disk_checkpoint, 'state_dict', disk_checkpoint)
        for name, value in checkpoint.items():
            if ('quant_activation' in name or 'quant_weight' in name) and name.split('.')[-1] in args.verbose_list:
                print(name, value.shape, value.requires_grad)
                print(value.data)
            elif "all" in args.verbose_list:
                if 'num_batches_tracked' not in name:
                    print(name, value.shape, value.requires_grad)

    if 'load' in config.keys():
        model_name = args.model
        if model_name in models.model_zoo:
            model, args = models.get_model(args)
        else:
            print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
            return
        if utils.check_file(args.pretrained):
            raw = 'raw' in config.keys()
            checkpoint = torch.load(args.pretrained)
            try:
                utils.load_state_dict(model, checkpoint.get('state_dict', None) if not raw else checkpoint, verbose=True)
            except RuntimeError:
                print("Loading pretrained model failed")
            print("Loading pretrained model OK")
        else:
            print("file not exist %s" % args.pretrained)

    if 'update' in config.keys():
        mapping_from = []
        mapping_to = []
        if os.path.isfile(args.mapping_from):
            with open(args.mapping_from) as f:
                mapping_from = f.readlines()
                f.close()
        if os.path.isfile(args.mapping_to):
            with open(args.mapping_to) as f:
                mapping_to = f.readlines()
                f.close()
        mapping_from = [ i.strip().strip('\n').strip('"').strip("'") for i in mapping_from]
        mapping_from = [ i for i in mapping_from if len(i) > 0 and i[0] != '#'] 
        mapping_to = [ i.strip().strip('\n').strip('"').strip("'") for i in mapping_to]
        mapping_to = [ i for i in mapping_to if len(i) > 0 and i[0] != '#']
        if len(mapping_to) != len(mapping_from) or len(mapping_to) == 0 or len(mapping_from) == 0:
            mapping = None
            logging.info('no valid mapping')
        else:
            mapping = {k: mapping_to[i] for i, k in enumerate(mapping_from) }

        raw = 'raw' in config.keys()
        if not os.path.isfile(args.old):
            args.old = args.pretrained
        utils.import_state_dict(args.old, args.new, mapping, raw, raw_prefix=args.case)

    if 'replace' in config.keys():
        mapping_from = []
        mapping_to = []
        if os.path.isfile(args.mapping_from):
            with open(args.mapping_from) as f:
                mapping_from = f.readlines()
                f.close()
        mapping_from = [ i.strip().strip('\n').strip('"').strip("'") for i in mapping_from]
        mapping_from = [ i for i in mapping_from if len(i) > 0] 
        with open(args.mapping_to, 'w') as f:
            for line in mapping_from:
                if 'weight_clip_value' in line:
                    line = line.replace('weight_clip_value', 'quant_weight.clip_val')
                if 'activation_clip_value' in line:
                    line = line.replace('activation_clip_value', 'quant_activation.clip_val')
                f.write(line + "\n")
            f.close()

    if 'det-load' in  config.keys():
        from third_party.checkpoint import DetectionCheckpointer
        model_name = args.model
        if model_name in models.model_zoo:
            model, args = models.get_model(args)
        else:
            print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
            return
        split = os.path.split(args.pretrained)
        checkpointer = DetectionCheckpointer(model, split[0], save_to_disk=True)
        checkpointer.resume_or_load(args.pretrained, resume=True)
        checkpointer.save(split[1])

    if 'mobile' in config.keys():
        utils.import_mobile(args)

    if 'verify-data' in config.keys() or 'verify-image' in config.keys():
        if 'verify-image' in config.keys():
            lists = args.verbose_list
        else:
            with open(os.path.join(args.root, 'train.txt')) as f:
                lists = f.readlines()
                f.close()
        from PIL import Image
        from threading import Thread
        print("going to check %d files" % len(lists))
        def check(lists, start, end, index):
            for i, item in enumerate(lists[start:end]):
                try:
                    items = item.split()
                    if len(items) >= 1:
                        path = items[0].strip().strip('\n')
                    else:
                        print("skip line %s" % i)
                        continue
                    path = os.path.join(args.root, os.path.join("train", path))
                    imgs = Image.open(path)
                    imgs.resize((256,256))
                    if index == 0:
                        print(i, end ="\r", file=sys.stderr)
                except (RuntimeError, IOError):
                    print("\nError when read image %s" % path)
            print("\nFinish checking", index)
        #lists = lists[45000:]
        num = min(len(lists), 20)
        for i in range(num):
            start = len(lists) // num * i
            end = min(start + len(lists) // num, len(lists))
            th = Thread(target=check, args=(lists, start, end, i))
            th.start()

    if 'lr_test' in config.keys():
      epoch = 0
      while epoch < args.epochs:
          lr = utils.adjust_learning_rate(None, epoch, args)
          print('[epoch %d]: lr %e' %(epoch, lr))
          epoch += 1

if __name__ == '__main__':
    #from XNorNet.ImageNet.networks.util import BinOp
    main()

