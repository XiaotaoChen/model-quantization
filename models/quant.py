
import math
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

if sys.version_info[0] == 3:
    from . import alqnet as alqnet
    from . import dorefa as dorefa
    from . import xnor as xnor

__EPS__ = 0 #1e-5

class quantization(nn.Module):
    def __init__(self, args=None, tag='fm', shape=[], feature_stride=None, logger=None):
        super(quantization, self).__init__()
        if args is None:
            return

        self.args = args
        self.tag = tag
        self.shape = shape
        self.feature_stride = feature_stride
        self.enable = getattr(args, tag + '_enable', False)
        self.adaptive = getattr(self.args, self.tag + '_adaptive', 'none')
        self.grad_scale = getattr(self.args, self.tag + '_grad_scale', 'none')
        self.custom = getattr(args, tag + '_custom', 'none')
        self.bit = getattr(args, tag + '_bit', None)
        self.num_levels = getattr(args, tag + '_level', None)
        self.half_range = getattr(args, tag + '_half_range', None)
        self.scale = getattr(args, tag + '_scale', 0.5)
        self.ratio = getattr(args, tag + '_ratio', 1)
        self.correlate = getattr(args, tag + '_correlate', -1)
        self.quant_group = getattr(args, tag + '_quant_group', None)
        self.boundary = getattr(self.args, self.tag + '_boundary', None)
        if self.bit is None:
            self.bit = 32
        if self.num_levels is None or self.num_levels <= 0:
            self.num_levels = int(2 ** self.bit)
        self.bit = (int)(self.bit)
        if self.half_range is None:
            self.half_range = tag == 'fm'
        else:
            self.half_range = bool(self.half_range)

        if self.quant_group == 0:
            self.quant_group = None
        if self.quant_group is not None:
            if self.quant_group < 0:
                if (shape[0] * shape[1]) % (-self.quant_group) != 0:
                    self.quant_group = None
                else:
                    self.quant_group = (shape[0] * shape[1]) / (-self.quant_group)
            else:
                if (shape[0] * shape[1]) % self.quant_group != 0:
                    self.quant_group = None
        if self.quant_group is not None:
            self.quant_group = int(self.quant_group)
        else:
            # layer wise for feature map, channel wise for weight
            self.quant_group = shape[0] if self.tag == 'wt' else 1
            ## channel wise for both
            #self.quant_group = shape[0] if self.tag == 'wt' else shape[1]

        self.fan = 1 # mode = 'fan_in' as default 
        for i in range(len(self.shape)-1):
            self.fan *= self.shape[i+1]

        if 'proxquant' in self.args.keyword:
            self.prox = 0

        if not self.enable:
            return

        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
        self.logger.info("half_range({}), bit({}), num_levels({}), quant_group({}) boundary({}) scale({}) ratio({}) fan({}) tag({})".format(
            self.half_range, self.bit, self.num_levels, self.quant_group, self.boundary, self.scale, self.ratio, self.fan, self.tag))

        self.method = 'none'
        self.times = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.learning_rate = 1
        self.init_learning_rate = 1
        self.progressive = False
        self.index = -1
        self.init()

    def init(self):
        # for LQ-Net
        if 'lq' in self.args.keyword or 'alq' in self.args.keyword or 'popcount' in self.args.keyword:
            if not hasattr(self, 'num_levels'):
                self.num_levels = 2**self.bit
            if self.num_levels > 256:
                raise RuntimeError("currently not support more than 8 bit quantization")
            if self.num_levels == 3:
                self.bit = 1
                self.logger.info('update %s_bit %r' % (self.tag, self.bit))

            self.method = 'lqnet'

            if 'lq' in self.args.keyword:
                self.lq_net_init()
                self.quant_fm = alqnet.LqNet_fm
                self.quant_wt = alqnet.LqNet_wt

            # initialize rould threshold
            init_thrs_multiplier = []
            for i in range(1, self.num_levels):
                thrs_multiplier_i = [0. for j in range(self.num_levels)]
                if not self.half_range:
                    if i < self.num_levels/2:
                        thrs_multiplier_i[i - 1] = 1 - self.scale
                        thrs_multiplier_i[i] = self.scale
                    elif i > self.num_levels/2:
                        thrs_multiplier_i[i - 1] = self.scale
                        thrs_multiplier_i[i] = 1 - self.scale
                    else:
                        thrs_multiplier_i[i - 1] = 0.5
                        thrs_multiplier_i[i] = 0.5
                else:
                    thrs_multiplier_i[i - 1] = self.scale
                    thrs_multiplier_i[i] = 1 - self.scale
                init_thrs_multiplier.append(thrs_multiplier_i)

            self.thrs_multiplier = nn.Parameter(torch.zeros(self.num_levels - 1, self.num_levels), requires_grad=False)
            self.thrs_multiplier.data = torch.FloatTensor(init_thrs_multiplier)
            if 'debug' in self.args.keyword:
                self.logger.info('self.thrs_multiplier: {}'.format(self.thrs_multiplier))

        if 'dorefa' in self.args.keyword or 'pact' in self.args.keyword:
            self.method = 'dorefa'
            if self.boundary is None:
                self.boundary = 1.0
                self.logger.info('update %s_boundary %r' % (self.tag, self.boundary))
            if self.tag == 'fm':
                if 'pact' in self.args.keyword:
                    self.quant_fm = dorefa.qfn
                    self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                elif 'lsq' in self.args.keyword or 'fm_lsq' in self.args.keyword:
                    self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                    self.quant_fm = dorefa.LSQ
                elif 'non-uniform' in self.args.keyword:
                    self.clip_val = nn.Parameter(torch.Tensor([self.boundary]), requires_grad = False)
                    self.custom_ratio = self.ratio
                    self.quant_fm = dorefa.RoundSTE
                    assert self.num_levels <= 4, 'non-uniform target at 2bit, ter, bin'
                    assert self.half_range or self.num_levels == 3, 'Full range quantization for activation supports ternary only'
                    for i in range(self.num_levels-1):
                        setattr(self, "alpha%d" % i, nn.Parameter(torch.ones(1)))
                        getattr(self, "alpha%d" % i).data.fill_(self.scale / self.boundary)
                    if 'gamma' in self.args.keyword:
                        self.basis = nn.Parameter(torch.ones (1), requires_grad=False)
                        self.auxil = nn.Parameter(torch.zeros(1), requires_grad=False)
                else: # Dorefa-Net
                    self.quant_fm = dorefa.qfn
                    self.clip_val = self.boundary
            else:
                if 'lsq' in self.args.keyword or 'wt_lsq' in self.args.keyword:
                    if self.shape[0] == 1:  ## linear
                        raise RuntimeError("Quantization for linear layer not provided")
                    else:
                        self.clip_val = nn.Parameter(torch.zeros(self.quant_group, 1, 1, 1))
                    self.clip_val.data.fill_(self.boundary)
                    self.quant_wt = dorefa.LSQ
                elif 'non-uniform' in self.args.keyword:
                    self.quant_wt = dorefa.RoundSTE
                    self.custom_ratio = self.ratio
                    assert self.num_levels == 3, 'non-uniform quantization for weight targets at ter'
                    for i in range(self.num_levels-1):
                        setattr(self, "alpha%d" % i, nn.Parameter(torch.ones(self.quant_group, 1, 1, 1)))
                        getattr(self, "alpha%d" % i).data.mul_(self.scale)
                    if 'debug' in self.args.keyword:
                        self.logger.info('debug: tag: {}, enter non-uniform'.format(self.tag))
                elif 'wt_bin' in self.args.keyword and self.num_levels == 2:
                    self.quant_wt = dorefa.DorefaParamsBinarizationSTE
                else:
                    self.quant_wt = dorefa.qfn
                    self.clip_val = self.boundary

        if 'xnor' in self.args.keyword:
            if self.tag == 'fm':
                self.quant_fm = xnor.XnorActivation
                if 'debug' in self.args.keyword:
                    self.logger.info('debug: tag: {} custom: {}'.format(self.tag, self.custom))
            else:
                self.quant_wt = xnor.XnorWeight
                if 'gamma' in self.args.keyword:
                    self.gamma = nn.Parameter(torch.ones(self.quant_group, 1, 1, 1))

        if 'ttn' in self.args.keyword:
            self.quant_group = 1 # consirder channel wise quant later
            self.ttn_init()
            self.method = 'ttn'

        #raise RuntimeError("Quantization method not provided %s" % self.args.keyword)

    def update_quantization(self, **parameters):
        if not self.enable:
            return

        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.logger.info('update %s_index %r' % (self.tag, self.index))

        if self.method == 'dorefa':
            if 'progressive' in parameters:
                self.progressive = parameters['progressive']
                self.logger.info('update %s_progressive %r' % (self.tag, self.progressive))

            if self.progressive:
                bit = self.bit
                num_level = self.num_levels
                if self.tag == 'fm':
                    if 'fm_bit' in parameters:
                        bit = parameters['fm_bit']
                    if 'fm_level' in parameters:
                        num_level = parameters['fm_level']
                else:
                    if 'wt_bit' in parameters:
                        bit = parameters['wt_bit']
                    if 'wt_level' in parameters:
                        num_level = parameters['wt_level']

                if bit != self.bit:
                    self.bit = bit
                    num_level = 2**self.bit
                    self.logger.info('update %s_bit %r' % (self.tag, self.bit))
                if num_level != self.num_levels:
                    self.num_levels = num_level
                    self.logger.info('update %s_level %r' % (self.tag, self.num_levels))

        if self.method == 'lqnet':
            pass

    def init_based_on_warmup(self, data=None):
        with torch.no_grad():
            if self.method == 'dorefa' and data is not None:
                pass
        return

    def init_based_on_pretrain(self, weight=None):
        with torch.no_grad():
            if self.method == 'dorefa' and 'non-uniform' in self.args.keyword:
                pass
        return

    def update_bias(self, basis=None):
        if not self.training:
            return

        if 'custom-update' not in self.args.keyword:
            self.basis.data = basis
            self.times.data = self.times.data + 1
        else:
            self.basis.data = self.basis.data * self.times  + self.auxil
            self.times.data = self.times.data + 1
            self.basis.data = self.basis.data / self.times

    def quantization_value(self, x, y):
        if self.times.data < self.args.stable:
            self.init_based_on_warmup(x)
            return x
        elif 'proxquant' in self.args.keyword:
            return x * self.prox + y * (1 - self.prox)
        else:
            return y

    def forward(self, x):
        if not self.enable:
            return x

        if self.method == 'lqnet':
            if self.tag == 'fm':
                y, basis = self.quant_fm.apply(x, self.basis, self.codec_vector, self.codec_index, self.thrs_multiplier, \
                        self.training, self.half_range, self.auxil, self.adaptive)
            else:
                y, basis = self.quant_wt.apply(x, self.basis, self.codec_vector, self.codec_index, self.thrs_multiplier, \
                        self.training, self.half_range, self.auxil, self.adaptive)

            self.update_bias(basis)

            return self.quantization_value(x, y)

        if self.method == 'ttn':
            return TTN.apply(x, self.wp, self.wn, self.thre)

        if 'xnor' in self.args.keyword:
            if self.tag == 'fm':
                y = self.quant_fm.apply(x, self.custom)
            else:
                if self.adaptive == 'var-mean':
                    std, mean = torch.std_mean(x.data.reshape(self.quant_group, -1, 1, 1, 1), 1)
                    x = (x - mean) / (std + __EPS__)
                y = self.quant_wt.apply(x)
                if 'gamma' in self.args.keyword:
                    y = y * self.gamma

            return self.quantization_value(x, y)

        if self.method == 'dorefa':
            if self.tag == 'fm':
                if 'lsq' in self.args.keyword or 'fm_lsq' in self.args.keyword:
                    if self.half_range:
                        y = x / self.clip_val
                        y = torch.clamp(y, min=0, max=1)
                        y = self.quant_fm.apply(y, self.num_levels - 1)
                        y = y * self.clip_val
                    else:
                        y = x / self.clip_val
                        y = torch.clamp(y, min=-1, max=1)
                        y = (y + 1.0) / 2.0
                        y = self.quant_fm.apply(y, self.num_levels - 1)
                        y = y * 2.0 - 1.0
                        y = y * self.clip_val
                elif 'pact' in self.args.keyword:
                    # note: we don't clip on the lower boundary, add F.ReLU if required
                    # the gradient of clip_val is automatically calculated by the torch.where
                    y = torch.where(x < self.clip_val, x, self.clip_val)
                    y = self.quant_fm.apply(y, self.num_levels, self.clip_val, self.adaptive)
                elif 'non-uniform' in self.args.keyword:
                    if self.half_range:
                        y1 = x * self.alpha0
                        y1 = torch.clamp(y1, min=0, max=1)
                        y1 = self.quant_fm.apply(y1, self.custom_ratio)
                        y = y1
                        if self.num_levels >= 3:
                            y2 = (x - 1.0/self.alpha0) * self.alpha1
                            y2 = torch.clamp(y2, min=0, max=1)
                            y2 = self.quant_fm.apply(y2, self.custom_ratio)
                            y = y + y2
                        if self.num_levels == 4:
                            y3 = (x - (1.0/self.alpha0 + 1.0/self.alpha1)) * self.alpha2
                            y3 = torch.clamp(y3, min=0, max=1)
                            y3 = self.quant_fm.apply(y3, self.custom_ratio)
                            y =  y + y3
                    else:
                        y1 = x * self.alpha0
                        y1 = torch.clamp(y1, min=-1, max=0)
                        y1 = self.quant_fm.apply(y1, self.custom_ratio)
                        y2 = x * self.alpha1
                        y2 = torch.clamp(y2, min=0, max=1)
                        y2 = self.quant_fm.apply(y2, self.custom_ratio)
                        y = y1 + y2
                    if 'gamma' in self.args.keyword:
                        if self.training:
                            self.auxil.data = dorefa.non_uniform_scale(x.detach(), y.detach())
                            self.update_bias(self.auxil.data)
                        y = y * self.basis
                else: # default dorefa
                    y = torch.clamp(x, min=0, max=self.clip_val)
                    y = self.quant_fm.apply(y, self.num_levels, self.clip_val, self.adaptive)
            else:
                if self.adaptive == 'var-mean':
                    std, mean = torch.std_mean(x.data.reshape(self.quant_group, -1, 1, 1, 1), 1)
                    x = (x - mean) / (std + __EPS__)
                if 'lsq' in self.args.keyword or 'wt_lsq' in self.args.keyword:
                    y = x / self.clip_val
                    y = torch.clamp(y, min=-1, max=1)
                    y = (y + 1.0) / 2.0
                    y = self.quant_wt.apply(y, self.num_levels - 1)
                    y = y * 2.0 - 1.0
                    y = y * self.clip_val
                elif 'non-uniform' in self.args.keyword:
                    y1 = x * self.alpha0
                    y1 = torch.clamp(y1, min=-1, max=0)
                    y1 = self.quant_wt.apply(y1, self.custom_ratio)
                    y2 = x * self.alpha1
                    y2 = torch.clamp(y2, min=0, max=1)
                    y2 = self.quant_wt.apply(y2, self.custom_ratio)
                    y = y1 + y2
                elif 'wt_bin' in self.args.keyword and self.num_levels == 2:
                    y = self.quant_wt.apply(x, self.adaptive)
                else:
                    y = torch.tanh(x)
                    y = y / (2 * y.abs().max()) + 0.5
                    y = 2 * self.quant_wt.apply(y, self.num_levels, self.clip_val, self.adaptive) - 1

            self.times.data = self.times.data + 1
            return self.quantization_value(x, y)

        raise RuntimeError("Should not reach here in quant.py")

    def ttn_init(self):
        self.wp = nn.Parameter(torch.ones(self.quant_group, 1), requires_grad=True)
        self.wn = nn.Parameter(torch.ones(self.quant_group, 1), requires_grad=True)
        self.thre = nn.Parameter(torch.ones(self.quant_group, 1), requires_grad=False)
        self.thre.data.fill_(getattr(self.args, self.tag + '_separator', 0.05))

    def lq_net_init(self):
        self.basis = nn.Parameter(torch.ones(self.bit, self.quant_group), requires_grad=False)
        self.auxil = nn.Parameter(torch.zeros(self.bit, self.quant_group), requires_grad=False)
        self.codec_vector = nn.Parameter(torch.ones(self.num_levels, self.bit), requires_grad=False)
        self.codec_index = nn.Parameter(torch.ones(self.num_levels, dtype=torch.int), requires_grad=False)

        init_basis = []
        NORM_PPF_0_75 = 0.6745
        if self.tag == 'fm':
            base = NORM_PPF_0_75 * 2. / (2 ** (self.bit - 1))
        else:
            base = NORM_PPF_0_75 * ((2. / self.fan) ** 0.5) / (2 ** (self.bit - 1))
        for i in range(self.bit):
            init_basis.append([(2 ** i) * base for j in range(self.quant_group)])
        self.basis.data = torch.FloatTensor(init_basis)

        # initialize level_codes
        init_level_multiplier = []
        for i in range(self.num_levels):
            level_multiplier_i = [0. for j in range(self.bit)]
            level_number = i
            for j in range(self.bit):
                binary_code = level_number % 2
                if binary_code == 0 and not self.half_range:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.codec_vector.data = torch.FloatTensor(init_level_multiplier)

        init_codec_index = []
        for i in range(self.num_levels):
            init_codec_index.append(i)
        self.codec_index.data= torch.IntTensor(init_codec_index)

class custom_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, args=None, force_fp=False, feature_stride=1):
        super(custom_conv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.args = args
        self.force_fp = force_fp
        if not self.force_fp:
            self.pads = padding
            self.padding = (0, 0)
            self.quant_activation = quantization(args, 'fm', [1, in_channels, 1, 1], feature_stride=feature_stride)
            self.quant_weight = quantization(args, 'wt', [out_channels, in_channels, kernel_size, kernel_size])
            self.padding_after_quant = getattr(args, 'padding_after_quant', False) if args is not None else False

    def init_after_load_pretrain(self):
        if not self.force_fp:
            self.quant_weight.init_based_on_pretrain(self.weight.data)
            self.quant_activation.init_based_on_pretrain()

    def update_quantization_parameter(self, **parameters):
        if not self.force_fp:
            self.quant_activation.update_quantization(**parameters)
            self.quant_weight.update_quantization(**parameters)

    def forward(self, inputs):
        if not self.force_fp:
            weight = self.quant_weight(self.weight)
            if self.padding_after_quant:
                inputs = self.quant_activation(inputs)
                inputs = F.pad(inputs, _quadruple(self.pads), 'constant', 0)
            else: # ensure the correct quantization levels (for example, BNNs only own the -1 and 1. zero-padding should be quantized into one of them
                inputs = F.pad(inputs, _quadruple(self.pads), 'constant', 0)
                inputs = self.quant_activation(inputs)
        else:
            weight = self.weight

        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

def conv5x5(in_planes, out_planes, stride=1, groups=1, args=None, force_fp=False, feature_stride=1):
    "5x5 convolution with padding"
    return custom_conv(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, groups=groups,
            args=args, force_fp=force_fp, feature_stride=feature_stride)

def conv3x3(in_planes, out_planes, stride=1, groups=1, args=None, force_fp=False, feature_stride=1):
    "3x3 convolution with padding"
    return custom_conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups,
            args=args, force_fp=force_fp, feature_stride=feature_stride)

def conv1x1(in_planes, out_planes, stride=1, args=None, force_fp=False, feature_stride=1):
    "1x1 convolution"
    return custom_conv(in_planes, out_planes, kernel_size=1, stride=stride, args=args,
            force_fp=force_fp, feature_stride=feature_stride)

#class custom_linear(nn.Linear):
#    def __init__(self, in_channels, out_channels, dropout=0, args=None, bias=False):
#        super(custom_linear, self).__init__(in_channels, out_channels, bias=bias)
#        self.args = args
#        self.dropout = dropout
#        self.quant_activation = quantization(args, 'fm', [1, in_channels, 1, 1])
#        self.quant_weight = quantization(args, 'wt', [1, 1, out_channels, in_channels])
#
#    def init_after_load_pretrain(self):
#        self.quant_weight.init_based_on_pretrain(self.weight.data)
#        self.quant_activation.init_based_on_pretrain()
#
#    def update_quantization_parameter(self, epoch=0, length=0):
#        self.quant_activation.update_quantization(epoch, length)
#        self.quant_weight.update_quantization(epoch, length)
#
#    def forward(self, inputs):
#        weight = self.quant_weight(self.weight)
#        inputs = self.quant_activation(inputs)
#        output = F.linear(inputs, weight, self.bias)
#        if self.dropout != 0:
#            output = F.dropout(output, p=self.dropout, training=self.training)
#
#        return output
#
#def qlinear(in_planes, out_planes, dropout=0, args=None):
#    "1x1 convolution"
#    return custom_linear(in_planes, out_planes, dropout=dropout, args=args)

