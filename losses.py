#!/usr/bin/python
# author eson
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAbsoluteError(nn.Module):
    '''


    '''

    def __init__(self):
        super(MeanAbsoluteError, self).__init__()

    def forward(self,logits, labels):
        if len(labels.shape)==1:
            n = logits.shape[0]
            one_hot_labels = torch.zeros_like(logits).scatter_(1, labels.reshape(n, -1), 1)
            sample_sum_error=torch.mean(torch.abs(logits - one_hot_labels),dim=1)

        return torch.mean(sample_sum_error,dim=0)

class ImprovedMeanAbsoluteError(MeanAbsoluteError):
    def __init__(self):
        super(ImprovedMeanAbsoluteError, self).__init__()
    def forward(self, y_pred, y_true):
        pass

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

if __name__ == '__main__':
    y_true = torch.tensor([1,1,0])
    y_pred = torch.tensor([[0.2, 0.8], [0, 1], [0.5, 0.5]])
    mae = MeanAbsoluteError()
    loss1=mae(y_pred, y_true)
    print(loss1)

    l1loss=nn.L1Loss(reduction='mean')
    # loss2=l1loss(y_pred,y_true)
    # print(loss2)