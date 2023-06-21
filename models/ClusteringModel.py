#!/usr/bin/python
# author eson

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveModel(nn.Module):
    pass

class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1, transform=None):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

        #
        self.model_transform = transform
        #

    def forward(self, x, forward_pass='default'):

        #
        if self.model_transform:
            x = self.model_transform(x)
        #

        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out


class OneHeadClusteringModel(nn.Module):
    def __init__(self, clustering_model):
        super(OneHeadClusteringModel, self).__init__()
        self.backbone = clustering_model.backbone
        self.backbone_dim = clustering_model.backbone_dim
        self.cluster_head = clustering_model.cluster_head
        self.model_transform = clustering_model.model_transform

    def forward(self, x, forward_pass='default'):

        #
        if self.model_transform:
            x = self.model_transform(x)
        #

        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out[0]


class SupervisedModel(nn.Module):
    pass


if __name__ == '__main__':
    pass