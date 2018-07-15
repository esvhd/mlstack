import torch
import torch.nn as nn


class LockedDropout(nn.Module):

    def __init__(self):
        """
        Based Stephen merity's awd-lstm-lm.
        """
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x

        # create tensor like x with all values set to 1 - dropout
        # generate bernoulli masks with 1 - dropout probability
        m = (torch.zeros(1, x.size(1), x.size(2)) + (1 - dropout)).bernoulli()
        mask = m / (1 - dropout)

        mask = mask.expand_as(x)

        if x.is_cuda:
            mask = mask.cuda()

        return mask * x
