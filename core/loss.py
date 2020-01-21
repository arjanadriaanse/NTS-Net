import torch
from torch import autograd, nn
from torch.nn.modules import loss
import torch.nn.functional as F

from core import gaussian_kernel

###
### The CrossEntropyLoss from torch.nn.modules.loss.
###
class CrossEntropyLoss(loss._WeightedLoss): 
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return self.cross_entropy(input=input, target=target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
    
    ###
    ### The cross_entropy function from toch.nn.functional,
    ### uses nll_loss from torch.nn.functional, the negative log likelihood function
    ### Using log_softmax in stead of softmax is recommended since log_softmax is numerically more stable.
    ### from: https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727
    ###
    def cross_entropy(self, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)
        return F.nll_loss(F.log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)

class CustomLoss(loss._WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']
    
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CustomLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return self.softmax_cross_entropy_with_logits(input=input, target=target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

    
    def softmax_cross_entropy_with_logits(self, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        return self.correntrophy(input, target, weight, None, ignore_index, None, reduction)
        ### Based on a TensorFlow loss function softmax_cross_entropy_with_logits, always 0.
        return F.log_softmax(F.nll_loss(F.log_softmax(input, 1), target, weight, None, ignore_index, None, reduction), -1)
        ### Correntrophy loss function, no results higher than 5% accuracy have been recorded.
        return F.nll_loss(input, target, weight, None, ignore_index, None, reduction).exp()

    def correntrophy(self, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        result = []
        for y_true, y_pred in zip(input, output):
            result.add(calculate(y_true, y_pred))
        return result
    
    def calculate(y_true, y_pred):
        - self.k_sum(self.robust_kernel(y_pred - y_true))
    
    def robust_kernel(tensor, sigma):
        return 1 / (sqrt(2 * pi * sigma)) * self.k_exp(-self.k_square(tensor) / (2 * sigma * sigma))
    
    def k_square(tensor):
        return tensor ** 2
    
    def k_exp(tensor):
        return torch.exp(tensor)
    
    def k_sum(tensor):
        return torch.sum(tensor)

#creterion = CrossEntropyLoss()
#print(creterion.correntrophy([0,0,0,0],[0,0,0,0])