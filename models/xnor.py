
import torch

##
class XnorActivation(torch.autograd.Function):
    '''
    Binarize the activations and calculate the mean across channel / resolution dimension.
    '''
    @staticmethod
    def forward(self, x, reduce_type='channel'):
        self.save_for_backward(x)
        b, c, h, w = x.shape
        if reduce_type == 'resolution':
            E = x.detach().abs().reshape(b, c, -1).mean(2, keepdim=True).reshape(b, c, 1, 1)
        elif reduce_type == 'channel':
            E = x.detach().abs().mean(1, keepdim=True)
        else:
            E = 1 # avoid runtime cost to calculate mean
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)
        y.mul_(E)
        return y

    @staticmethod
    def backward(self, grad_output):
        x, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.ge(1)] = 0
        grad_input[x.le(-1)] = 0
        return grad_input, None

## 
class XnorWeight(torch.autograd.Function):
    '''
    Binarize the weight
    '''
    @staticmethod
    def forward(ctx, x, quant_group=1):
        E = x.detach().abs().reshape(quant_group, -1).mean(1, keepdim=True).reshape(quant_group, 1, 1, 1)
        y = torch.ones_like(x)
        y.masked_fill_(x < 0, -1)  # use compare rather than sign() to handle the zero
        y.mul_(E)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

