import torch
from torch import autograd, nn
from torch.nn.modules import loss
import torch.nn.functional as F

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
    ### The cross_entropy function based on toch.nn.functional,
    ### uses nll_loss from torch.nn.functional, the negative log likelihood function
    ###
    def cross_entropy(self, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)
        # return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
        return F.nll_loss(input, target, weight, None, ignore_index, None, reduction)

"""
cross_entropy loss function defined in tensor flow
tf.losses.softmax_cross_entropy(
    onehot_labels,
    logits,
    weights = 1.0,
    label_smoothing = 0,
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES,
    reduction-Reduction.SUM_BY_NONZERO_WEIGHTS

labelsmoothing: smooth the labels towards 
1/num_classes: new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes

use BCEWithLogitsLoss? or adapt cross_entropy loss with BCEWithLogitsLoss

source : https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/10
best to implement your own.
cross entropy loss is something like this I think . . .
[0.1, 0.2, 0.7] (prediction) ------------------ [1.0, 0.0, 0.0] (target)
what you want is - (1.0 * log(0.1) + 0.0 * log(0.2) + 0.0 * log(0.7)) this is the cross entropy loss
so to translate that into code, you have prediction (a vector of length k) and target (a vector of length k, not nessesarily 1 hot)
what you would do would be something like -1 * sum(log(prediction) * target)
so this is what I have in my own code, hopefully it’s helpful

Application of softmax
https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch
"""










