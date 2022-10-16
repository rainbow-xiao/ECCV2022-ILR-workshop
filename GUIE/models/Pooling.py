import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM_Pooling(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM_Pooling,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=p_trainable)
        self.eps = eps
    
    def forward(self, x):
        out = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        return out
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'