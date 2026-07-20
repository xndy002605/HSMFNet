import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, centers=None, class_weights=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if centers is not None:
            self.centers = nn.Parameter(centers)
        else:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, x, labels):
        batch_size = x.size(0)
        device = x.device

        self.centers = self.centers.to(device)

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes, device=device, dtype=torch.long)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))

        if self.class_weights is not None:
            class_weights_expanded = self.class_weights[labels].unsqueeze(1).expand(batch_size, self.num_classes)
            dist = distmat * mask.float() * class_weights_expanded
        else:
            dist = distmat * mask.float()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss