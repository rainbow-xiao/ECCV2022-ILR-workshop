import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Smoth_CE_Loss(nn.Module):
    def __init__(self, ls_=0.9):
        super().__init__()
        self.crit = nn.CrossEntropyLoss(reduction="none")  
        self.ls_ = ls_

    def forward(self, logits, labels):
        labels *= self.ls_
        return self.crit(logits, labels)

class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=1)
        loss = -logprobs * target
        loss = loss.sum(dim=1)
        return loss

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.3, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        labels = F.one_hot(labels.long(), logits.shape[-1]).float()
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
    
class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, stride=0.1, max_m=0.8, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        if crit == "ce":
            self.crit = DenseCrossEntropy()
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.m = m
        self.m_s = stride
        self.max_m = max_m
        self.last_epoch = 1
        self.reduction = reduction
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
    def update(self, logger, c_epoch):
        self.m = min(self.m+self.m_s*(c_epoch-self.last_epoch), self.max_m)
        self.last_epoch = c_epoch
        logger.info('Update margin----')
        logger.info(f'Curent Epoch: {c_epoch}, Curent Margin: {self.m:.2f}')
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
    def forward(self, logger, logits, labels, mode='train', c_epoch=1):
        if c_epoch!=self.last_epoch and mode=='train':
            self.update(logger, c_epoch)
        cosine = logits.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type(cosine.type())
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        labels = F.one_hot(labels.long(), logits.shape[-1]).float()
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class Contrastive_Arc_Loss(nn.Module):
    def __init__(self, n_views=5, s=45.0, m=0.1, crit="ce", ls=0.9, reduction="mean"):
        super().__init__()
        self.n = n_views
        if crit == "ce":
            self.crit = DenseCrossEntropy()   
        elif crit == "smoth_ce":
            self.crit = Smoth_CE_Loss(ls_=ls)
        if s is None:
            self.s = nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, features):
        labels = torch.cat([torch.arange(features.shape[0]//self.n) for i in range(self.n)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        labels_positives = torch.ones_like(positives)
        labels_negatives = torch.zeros_like(negatives)
        cosine = torch.cat([positives, negatives], dim=1)
        labels = torch.cat([labels_positives, labels_negatives], dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss