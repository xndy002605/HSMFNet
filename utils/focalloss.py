import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class FocalLoss(_WeightedLoss):
    def __init__(self, alpha=None, gamma=2, label_smooth=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).to("cuda")
        self.label_smooth = label_smooth

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none',
                                  weight=self.alpha, label_smoothing=self.label_smooth)
        pred_logsoft = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-pred_logsoft)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss

if __name__ == "__main__":
    gt = torch.randint(0, 3, size=(1, ))
    pred = torch.randn((1, 3))
    pred.requires_grad = True
    weight = torch.Tensor([0.3, 0.6, 0.1])
    loss = FocalLoss(alpha=weight, gamma=2)
    loss_val = loss(pred, gt)
    print(gt.shape)
    print(pred.shape)
    print(gt)
    print(pred)
    print(f"focal loss of GT and pred is : {loss_val.item():.4f}")
    loss_val.backward()
    print("pred gradient after focal loss backward is \n", pred.grad)