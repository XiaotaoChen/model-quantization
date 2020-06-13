
import torch

__EPS__ = 1e-5

##########
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ratio=1.0):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

##########
class LSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

##############################################################
## Dorefa-net (https://arxiv.org/pdf/1606.06160.pdf)
class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_levels, clip_val=1.0, adaptive='none'):
        if adaptive == 'var':
            std = input.std()
            input = input / std

        n = float(num_levels - 1)
        scale = n / clip_val
        out = torch.round(input * scale) / scale
        if adaptive == 'var':
            out = out * std
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

## 
class DorefaParamsBinarizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, adaptive='none'):
        E = x.detach().abs().mean()
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)  # use compare rather than sign() to handle the zero
        y.mul_(E)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

## Xnor-Net
class Xnor(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(self, x):
        self.save_for_backward(x)
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)
        return y

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

## TTN (https://arxiv.org/pdf/1612.01064v1)
class TTN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wp, wn, thre):
        y = x.clone()
        quant_group = thre.size(0)
        y = y.reshape(quant_group, -1)
        thre_y = y.abs().max(dim=1, keepdim=True)[0] * thre
        thre_y = thre_y.expand_as(y)
        a = (y > thre_y).float()
        b = (y <-thre_y).float()
        y = a * wp - b * wn
        y = y.reshape(x.shape)
        ctx.save_for_backward(a, b, wp, wn)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        a, b, wp, wn = ctx.saved_tensors
        grad_out_shape = grad_out.shape
        grad_out = grad_out.reshape(a.shape)
        c = torch.ones_like(a) - a - b
        grad_wp = (a*grad_out).sum(dim=1, keepdim=True)
        grad_wn = (b*grad_out).sum(dim=1, keepdim=True)
        grad_in = (wp*a + wn*b* + 1.0*c) * grad_out
        return grad_in.reshape(grad_out_shape), grad_wp, grad_wn, None


## 
def non_uniform_scale(x, codec):
    BTxX = codec * x
    BTxX = BTxX.sum()
    BTXB = (codec * codec).sum()
    basis = BTxX / BTXB
    basis = torch.where(BTXB == 0, x.mean().to(torch.float32), basis)
    return basis

