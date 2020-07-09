import torch
import torch.nn as nn
import logging
import numpy as np

from .quant import conv3x3, conv1x1
from .layers import norm, actv
from .prone import qprone

# double_channel_half_resolution
class DCHR(nn.Module):
    def __init__(self, stride):
        super(DCHR, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=stride)

    def forward(self, x):
        pool = self.pool(x)
        shape = pool.shape
        shape = [i for i in shape]
        shape[1] = shape[1] // 2
        fill = x.new_zeros(shape)
        return torch.cat((fill, pool, fill), 1)

# TResNet: High Performance GPU-Dedicated Architecture (https://arxiv.org/pdf/2003.13630v1.pdf)
class TResNetStem(nn.Module):
    def __init__(self, out_channel, in_channel=3, stride=4, kernel_size=1, force_fp=True, args=None):
        super(TResNetStem, self).__init__()
        self.stride = stride
        assert kernel_size in [1, 3], "Error reshape conv kernel"
        if kernel_size == 1:
            self.conv = conv1x1(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)
        elif kernel_size == 3:
            self.conv = conv3x3(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        x = self.conv(x)
        return x

def seq_c_b_a_s(x, conv, relu, bn, skip, skip_enbale):
    out = conv(x)
    out = bn(out)
    out = relu(out)
    if skip_enbale:
        out += skip
    return out

def seq_c_b_s_a(x, conv, relu, bn, skip, skip_enbale):
    out = conv(x)
    out = bn(out)
    if skip_enbale:
        out += skip
    out = relu(out)
    return out

def seq_c_a_b_s(x, conv, relu, bn, skip, skip_enbale):
    out = conv(x)
    out = relu(out)
    out = bn(out)
    if skip_enbale:
        out += skip
    return out

def seq_b_c_a_s(x, conv, relu, bn, skip, skip_enbale):
    out = bn(x)
    out = conv(out)
    out = relu(out)
    if skip_enbale:
        out += skip
    return out

def seq_b_a_c_s(x, conv, relu, bn, skip, skip_enbale):
    out = bn(x)
    out = relu(out)
    out = conv(out)
    if skip_enbale:
        out += skip
    return out

'''
BasicBlock:
    different variants on architectures are supported (mainly controled by the order string
'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, args=None, feature_stride=1):
        super(BasicBlock, self).__init__()
        self.args = args

        # Bi-Real structure or original structure
        if 'origin' in args.keyword:
            self.addition_skip = False
        else:
            self.addition_skip = True

        if self.addition_skip and args.verbose:
            logging.info("warning: add addition skip, not the origin resnet")

        # quantize skip connection ?
        real_skip = 'real_skip' in args.keyword

        for i in range(2):
            setattr(self, 'relu%d' % (i+1), nn.ModuleList([actv(args) for j in range(args.base)]))
        if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
            self.fix_relu = actv(args)
            setattr(self, 'relu2', nn.ModuleList([nn.Sequential() for j in range(args.base)]))

        if 'cbas' in args.keyword:
            self.seq = seq_c_b_a_s
            order = 'cbas'
        elif 'cbsa' in args.keyword: # default architecture in Pytorch
            self.seq = seq_c_b_s_a
            order = 'cbsa'
        elif 'cabs' in args.keyword: # group-net
            self.seq = seq_c_a_b_s
            order = 'cabs'
        elif 'bacs' in args.keyword:
            self.seq = seq_b_a_c_s
            order = 'bacs'
        elif 'bcas' in args.keyword:
            self.seq = seq_b_c_a_s
            order = 'bcas'
        else:
            self.seq = None
            order = 'none'

        # lossless downsample network on
        self.order = getattr(args, "order", 'none')
        if 'ReShapeResolution' in args.keyword and stride != 1:
            shrink = []
            if self.order == 'none':
                self.order = order
            for i in self.order:
                if i == 'c':
                    shrink.append(TResNetStem(planes, in_channel=inplanes, stride=stride, args=args, force_fp=real_skip))
                if i == 'b':
                    if 'preBN' in args.keyword:
                        shrink.append(norm(inplanes, args))
                    else:
                        shrink.append(norm(planes, args))
                if i == 'a':
                    shrink.append(actv(args))
            self.shrink = nn.Sequential(*shrink)
            inplanes = planes
            stride = 1
        else:
            self.shrink = None
        # lossless downsample network off

        if 'bacs' in args.keyword or 'bcas' in args.keyword: 
            self.bn1 = nn.ModuleList([norm(inplanes, args, feature_stride=feature_stride) for j in range(args.base)])
            if 'fix' in self.args.keyword:
                self.fix_bn = norm(planes, args, feature_stride=feature_stride*stride)
        else:
            self.bn1 = nn.ModuleList([norm(planes, args, feature_stride=feature_stride*stride) for j in range(args.base)])
        self.bn2 = nn.ModuleList([norm(planes, args, feature_stride=feature_stride*stride) for j in range(args.base)])

        # Prone network on
        keepdim = True
        qconv3x3 = conv3x3
        qconv1x1 = conv1x1
        extra_padding = 0
        if 'prone' in args.keyword:
            keepdim = 'restore_shape' in args.keyword
            bn_before_restore = 'bn_before_restore' in args.keyword
            qconv3x3 = qprone

            if 'preBN' in args.keyword: # to be finished
                raise NotImplementedError("preBN not supported for the Prone yet")
            else:
                if not keepdim: # to be finished
                    self.bn1 = nn.ModuleList([norm(planes*4, args) for j in range(args.base)])
                    if bn_before_restore:
                        self.bn2 = nn.ModuleList([norm(planes*16, args) for j in range(args.base)])

            if stride != 1 and (args.input_size // feature_stride) % (2*stride) != 0:
                extra_padding = ((2*stride) - ((args.input_size // feature_stride) % (2*stride))) // 2
                logging.warning("extra pad for Prone is added to be {}".format(extra_padding))
        # Prone network off

        # downsample branch
        self.enable_skip = stride != 1 or inplanes != planes
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(stride))
        else:
            downsample.append(nn.Sequential())
        if inplanes != planes:
            if 'bacs' in args.keyword:
                downsample.append(norm(inplanes, args, feature_stride=feature_stride))
                downsample.append(actv(args))
                downsample.append(qconv1x1(inplanes, planes, stride=1, args=args, force_fp=real_skip, feature_stride=feature_stride*stride))
                if 'fix' in args.keyword:
                    downsample.append(norm(planes, args, feature_stride=feature_stride*stride))
            elif 'bcas' in args.keyword:
                downsample.append(norm(inplanes, args, feature_stride=feature_stride))
                downsample.append(qconv1x1(inplanes, planes, stride=1, args=args, force_fp=real_skip, feature_stride=feature_stride*stride))
                downsample.append(actv(args))
                if 'fix' in args.keyword: # remove the ReLU in skip connection
                    downsample.append(norm(planes, args, feature_stride=feature_stride*stride))
            else:
                downsample.append(qconv1x1(inplanes, planes, args=args, force_fp=real_skip, feature_stride=feature_stride*stride))
                downsample.append(norm(planes, args, feature_stride=feature_stride*stride))
                if 'fix' not in args.keyword:
                    downsample.append(actv(args))
        if 'singleconv' in args.keyword: # pytorch official branch employ single convolution layer
            for i, n in enumerate(downsample):
                if isinstance(n, nn.AvgPool2d):
                    downsample[i] = nn.Sequential()
                if isinstance(n, nn.Conv2d):
                    downsample[i] = qconv1x1(inplanes, planes, stride=stride, padding=extra_padding, args=args, force_fp=real_skip, feature_stride=feature_stride)
        if 'DCHR' in args.keyword: # try if any performance improvement when aligning resolution without downsample 
            if args.verbose:
                logging.warning("warning: DCHR is used in the block")
            self.skip = DCHR(stride)
        else:
            self.skip = nn.Sequential(*downsample)

        # second conv
        self.conv2 = nn.ModuleList([qconv3x3(planes, planes, 1, 1, args=args, feature_stride=feature_stride*stride) for j in range(args.base)])

        # first conv
        if 'prone' in args.keyword and 'no_prone_downsample' in args.keyword and stride != 1 and keepdim:
            qconv3x3 = conv3x3
        self.conv1 = nn.ModuleList([qconv3x3(inplanes, planes, stride, 1, padding=extra_padding+1, args=args, feature_stride=feature_stride, keepdim=keepdim) for j in range(args.base)])

        # scales
        if args.base == 1:
            self.scales = [1]
        else:
            self.scales = nn.ParameterList([nn.Parameter(torch.ones(1) / args.base, requires_grad=True) for i in range(args.base)])

        # Fixup initialization (https://arxiv.org/abs/1901.09321)
        if 'fixup' in args.keyword:
            self.bn1 = nn.ModuleList([nn.Sequential()])
            self.bn2 = nn.ModuleList([nn.Sequential()])
            for i, n in enumerate(self.skip):
                if isinstance(n, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.skip[i] = nn.Sequential()

            self.fixup_scale = nn.Parameter(torch.ones(1))
            if 'bias' in args.keyword:
                self.fixup_bias1a = nn.Parameter(torch.zeros(1))
                self.fixup_bias1b = nn.Parameter(torch.zeros(1))
                self.fixup_bias2a = nn.Parameter(torch.zeros(1))
                self.fixup_bias2b = nn.Parameter(torch.zeros(1))


    def forward(self, x):

        if self.shrink is not None:
            x = self.shrink(x)

        if not self.enable_skip:
            residual = x

        if 'fixup' in self.args.keyword:
            if 'bias' in self.args.keyword:
                x = x + self.fixup_bias1a

        if self.enable_skip:
            residual = self.skip(x)

        result = None
        for conv1, conv2, bn1, bn2, relu1, relu2, scale in zip(self.conv1, self.conv2, \
                self.bn1, self.bn2, self.relu1, self.relu2, self.scales):
            if 'fixup' in self.args.keyword and 'bias' in self.args.keyword:
                out = self.seq(x, conv1, relu1, bn1, self.fixup_bias1b, True) + self.fixup_bias2a
            else:
                out = self.seq(x, conv1, relu1, bn1, residual, self.addition_skip)
            output = self.seq(out, conv2, relu2, bn2, out, self.addition_skip)
            if result is None:
                result = scale * output
            else:
                result = result + scale * output
        output = result

        if 'fixup' in self.args.keyword:
            output = output * self.fixup_scale
            if 'bias' in self.args.keyword:
                output = output + self.fixup_bias2b

        if not self.addition_skip:
            if 'fix' in self.args.keyword and ('bacs' in self.args.keyword or 'bcas' in self.args.keyword):
                output = self.fix_bn(output)
            output = output + residual
            if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
                output = self.fix_relu(output)

        return output

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, args=None, feature_stride=1):
        super(BottleNeck, self).__init__()
        self.args = args

        # Bi-Real structure or original structure
        if 'origin' in args.keyword:
            self.addition_skip = False
        else:
            self.addition_skip = True

        if self.addition_skip and args.verbose:
                logging.info("warning: add addition skip, not the origin resnet")

        qconv3x3 = conv3x3
        qconv1x1 = conv1x1
        for i in range(3):
            setattr(self, 'relu%d' % (i+1), nn.ModuleList([actv(args) for j in range(args.base)]))
        if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
            setattr(self, 'relu3', nn.ModuleList([nn.Sequential() for j in range(args.base)]))
            self.fix_relu = actv(args)

        if 'cbas' in args.keyword:
            self.seq = seq_c_b_a_s
        elif 'cbsa' in args.keyword: # default Pytorch
            self.seq = seq_c_b_s_a
        elif 'cabs' in args.keyword: # group-net
            self.seq = seq_c_a_b_s
        elif 'bacs' in args.keyword:
            self.seq = seq_b_a_c_s
        elif 'bcas' in args.keyword:
            self.seq = seq_b_c_a_s
        else:
            self.seq = None

        if 'bacs' in args.keyword:
            self.bn1 = nn.ModuleList([norm(inplanes, args) for j in range(args.base)])
            self.bn3 = nn.ModuleList([norm(planes, args) for j in range(args.base)])
            if 'fix' in self.args.keyword:
                self.fix_bn = norm(planes * self.expansion, args)
        else:
            self.bn1 = nn.ModuleList([norm(planes, args) for j in range(args.base)])
            self.bn3 = nn.ModuleList([norm(planes * self.expansion, args) for j in range(args.base)])
        self.bn2 = nn.ModuleList([norm(planes, args) for j in range(args.base)])

        # downsample branch
        self.enable_skip = stride != 1 or inplanes != planes * self.expansion
        real_skip = 'real_skip' in args.keyword
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(stride))
        else:
            downsample.append(nn.Sequential())
        if inplanes != planes * self.expansion:
            if 'bacs' in args.keyword:
                downsample.append(norm(inplanes, args))
                downsample.append(actv(args))
                downsample.append(qconv1x1(inplanes, planes * self.expansion, stride=1, args=args, force_fp=real_skip, feature_stride=feature_stride*stride))
                if 'fix' in args.keyword:
                    downsample.append(norm(planes * self.expansion, args))
            else:
                downsample.append(qconv1x1(inplanes, planes * self.expansion, stride=1, args=args, force_fp=real_skip, feature_stride=feature_stride*stride))
                downsample.append(norm(planes * self.expansion, args))
                if 'fix' not in args.keyword:
                    downsample.append(actv(args))
        if 'singleconv' in args.keyword:
            for i, n in enumerate(downsample):
                if isinstance(n, nn.AvgPool2d):
                    downsample[i] = nn.Sequential()
                if isinstance(n, nn.Conv2d):
                    downsample[i] = qconv1x1(inplanes, planes * self.expansion, stride=stride, args=args, force_fp=real_skip, feature_stride=feature_stride)
        if 'DCHR' in args.keyword:
            if args.verbose:
                logging.info("warning: DCHR is used in the block")
            self.skip = DCHR(stride)
        else:
            self.skip = nn.Sequential(*downsample)

        self.conv1 = nn.ModuleList([qconv1x1(inplanes, planes, 1, args=args, feature_stride=feature_stride) for j in range(args.base)])
        self.conv2 = nn.ModuleList([qconv3x3(planes, planes, stride, 1, args=args, feature_stride=feature_stride) for j in range(args.base)])
        feature_stride = feature_stride * stride
        self.conv3 = nn.ModuleList([qconv1x1(planes, planes * self.expansion, 1, args=args, feature_stride=feature_stride) for j in range(args.base)])

        if args.base == 1:
            self.scales = [1]
        else:
            self.scales = nn.ParameterList([nn.Parameter(torch.ones(1) / args.base, requires_grad=True) for i in range(args.base)])

        if 'fixup' in args.keyword:
            assert args.base == 1, 'Base should be 1 in Fixup'
            self.bn1 = nn.ModuleList([nn.Sequential()])
            self.bn2 = nn.ModuleList([nn.Sequential()])
            self.bn3 = nn.ModuleList([nn.Sequential()])
            for i, n in enumerate(self.skip):
                if isinstance(n, (nn.BatchNorm2d, nn.GroupNorm)):
                    self.skip[i] = nn.Sequential()

            self.fixup_scale = nn.Parameter(torch.ones(1))
            if 'bias' in args.keyword:
                self.fixup_bias1a = nn.Parameter(torch.zeros(1))
                self.fixup_bias1b = nn.Parameter(torch.zeros(1))
                self.fixup_bias2a = nn.Parameter(torch.zeros(1))
                self.fixup_bias2b = nn.Parameter(torch.zeros(1))
                self.fixup_bias3a = nn.Parameter(torch.zeros(1))
                self.fixup_bias3b = nn.Parameter(torch.zeros(1))
            else:
                pass

    def forward(self, x):
        if not self.enable_skip:
            residual = x

        if 'fixup' in self.args.keyword:
            if 'bias' in self.args.keyword:
                x = x + self.fixup_bias1a

        if self.enable_skip:
            residual = self.skip(x)

        result = None
        for conv1, conv2, conv3, bn1, bn2, bn3, relu1, relu2, relu3, scale in zip(self.conv1, self.conv2, self.conv3, \
                    self.bn1, self.bn2, self.bn3, self.relu1, self.relu2, self.relu3, self.scales):
            if 'fixup' in self.args.keyword and 'bias' in self.args.keyword:
                out = self.seq(x, conv1, relu1, bn1, self.fixup_bias1b, True) + self.fixup_bias2a
                out = self.seq(out, conv2, relu2, bn2, self.fixup_bias2b, True) + self.fixup_bias3a
            else:
                out = self.seq(x, conv1, relu1, bn1, residual, self.addition_skip)
                out = self.seq(out, conv2, relu2, bn2, out, self.addition_skip)
            output = self.seq(out, conv3, relu3, bn3, out, self.addition_skip)
            if result is None:
                result = scale * output
            else:
                result = result + scale * output
        output = result

        if 'fixup' in self.args.keyword:
            output = output * self.fixup_scale
            if 'bias' in self.args.keyword:
                output = output + self.fixup_bias3b

        if not self.addition_skip:
            if 'fix' in self.args.keyword and ('bacs' in self.args.keyword or 'bcas' in self.args.keyword):
                output = self.fix_bn(output)
            output += residual
            if 'fix' in self.args.keyword and ('cbas' in self.args.keyword or 'cbsa' in self.args.keyword):
                output = self.fix_relu(output)

        return output

class ResNet(nn.Module):
    def __init__(self, block, layers, args):
        super(ResNet, self).__init__()
        self.args = args
        assert args is not None, "args is None"
        self.layer_count = len(layers)
        self.inplanes = 64
        self.width_alpha = getattr(args, 'width_alpha', 1.0)
        self.inplanes = int(self.inplanes * self.width_alpha)
        self.input_channel = self.inplanes
        self.feature_stride = 1

        if 'cifar10' in args.keyword or 'cifar100' in args.keyword:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Sequential()
        elif 'TResNetStem' in args.keyword or 'TResNetStemMaxPool' in args.keyword:
            if 'TResNetStemMaxPool' in args.keyword:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.conv1 = TResNetStem(self.input_channel, stride=2, kernel_size=args.stem_kernel)
            else:
                self.maxpool = nn.Sequential()
                self.conv1 = TResNetStem(self.input_channel, stride=4, kernel_size=args.stem_kernel)
            self.feature_stride = 4
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.feature_stride = 4

        outplanes = self.inplanes
        for i in range(self.layer_count):
            index = i + 1
            channel_scale = 2 ** i
            outplanes = self.input_channel * channel_scale
            stride = 1 if i == 0 else 2
            setattr(self, 'layer%d' % index, self._make_layer(block, outplanes, layers[i], stride=stride, feature_stride=self.feature_stride))
            self.feature_stride = self.feature_stride * stride

        if 'preBN' in args.keyword:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential(norm(outplanes * block.expansion, args))
            if 'patch_preBN_stem' in args.keyword:
                if 'fix_pooling' in self.args.keyword:
                    self.bn1 = nn.Sequential(norm(self.input_channel, args), actv(args))
                self.bn2 = nn.Sequential(norm(outplanes * block.expansion, args), actv(args))
        else:
            self.bn1 = nn.Sequential(norm(self.input_channel, args), actv(args))
            self.bn2 = nn.Sequential()
            if 'group-net' in args.keyword:
                self.bn1[1] = nn.Sequential()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(outplanes * block.expansion, args.num_classes)

        if 'debug' in args.keyword:
            logging.info("Resnet has attr '_out_features' %r" % hasattr(self, '_out_features'))
        if hasattr(self, '_out_features') and 'linear' not in self._out_features:
            self.avgpool = None
            self.fc = None
            self.bn2 = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if 'zir' in args.keyword:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    fix_bn = getattr(m, 'fix_bn', None)
                    if fix_bn is not None:
                        nn.init.constant_(m.fix_bn.weight, 0)
                    else:
                        for b in getattr(m, 'bn3', []):
                            nn.init.constant_(b.weight, 0)
                elif isinstance(m, BasicBlock):
                    fix_bn = getattr(m, 'fix_bn', None)
                    if fix_bn is not None:
                        nn.init.constant_(m.fix_bn.weight, 0)
                    else:
                        for b in getattr(m, 'bn2', []):
                            nn.init.constant_(b.weight, 0)

        if 'fixup' in args.keyword:
            self.stem_relu = actv(args)
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()
            if 'bias' in args.keyword:
                self.fixup_bias1 = nn.Parameter(torch.zeros(1))
                self.fixup_bias2 = nn.Parameter(torch.zeros(1))

            for m in self.modules():
                if isinstance(m, BasicBlock):
                    for i, n in enumerate(m.conv1):
                        nn.init.normal_(n.weight, mean=0, std=np.sqrt(2 / (n.weight.shape[0] * np.prod(n.weight.shape[2:]))) * self.layer_count ** (-0.5))
                    for i, n in enumerate(m.conv2):
                        nn.init.constant_(n.weight, 0)
                    for i, n in enumerate(m.skip):
                        if isinstance(n, nn.Conv2d):
                            nn.init.normal_(n.weight, mean=0, std=np.sqrt(2 / (n.weight.shape[0] * np.prod(n.weight.shape[2:]))))

                elif isinstance(m, BottleNeck):
                    for i, n in enumerate(m.conv1):
                        nn.init.normal_(n.weight, mean=0, std=np.sqrt(2 / (n.weight.shape[0] * np.prod(n.weight.shape[2:]))) * self.layer_count ** (-0.25))
                    for i, n in enumerate(m.conv2):
                        nn.init.normal_(n.weight, mean=0, std=np.sqrt(2 / (n.weight.shape[0] * np.prod(n.weight.shape[2:]))) * self.layer_count ** (-0.25))
                    for i, n in enumerate(m.conv3):
                        nn.init.constant_(n.weight, 0)
                    for i, n in enumerate(m.skip):
                        if isinstance(n, nn.Conv2d):
                            nn.init.normal_(n.weight, mean=0, std=np.sqrt(2 / (n.weight.shape[0] * np.prod(n.weight.shape[2:]))))

                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, feature_stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, self.args, feature_stride))
            feature_stride = feature_stride * stride
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}

        x = self.conv1(x)

        if 'fixup' in self.args.keyword:
            if 'bias' in self.args.keyword:
                x = self.stem_relu(x + self.fixup_bias1)
            else:
                x = self.stem_relu(x)

        if 'fix_pooling' in self.args.keyword:
            x = self.bn1(x)
            x = self.maxpool(x)
        else:
            x = self.maxpool(x)
            x = self.bn1(x)

        if hasattr(self, '_out_features') and 'stem' in self._out_features:
            outputs["stem"] = x

        for i in range(self.layer_count):
            layer = 'layer%d' % (i + 1)
            x = getattr(self, layer)(x)
            if hasattr(self, '_out_features') and layer in self._out_features:
                outputs[layer] = x

        if hasattr(self, '_out_features') and 'linear' not in self._out_features:
            return outputs
        
        #if 'keep_resolution' in self.args.keyword:
        #    B, C, H, W = x.shape
        #    if H == 8:
        #        x = x[:, :, 0:H-1, 0:W-1]

        x = self.bn2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if 'fixup' in self.args.keyword:
            if 'bias' in self.args.keyword:
                x = x + self.fixup_bias2

        x = self.fc(x)
        return x


def resnet18(args):
    model = ResNet(BasicBlock, [2, 2, 2, 2], args)
    return model

def resnet20(args):
    model = ResNet(BasicBlock, [3, 3, 3], args)
    return model

def resnet32(args):
    model = ResNet(BasicBlock, [5, 5, 5], args)
    return model

def resnet34(args):
    model = ResNet(BasicBlock, [3, 4, 6, 3], args)
    return model

def resnet44(args):
    model = ResNet(BasicBlock, [7, 7, 7], args)
    return model

def resnet50(args):
    model = ResNet(BottleNeck, [3, 4, 6, 3], args)
    return model

def resnet56(args):
    model = ResNet(BasicBlock, [9, 9, 9], args)
    return model

def resnet101(args):
    model = ResNet(BottleNeck, [3, 4, 23, 3], args)
    return model



