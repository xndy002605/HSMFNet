import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F


# =====================================================================
# CFL-HC: COARSE-FINE LINKED SUPERVISION HIERARCHICAL CLASSIFICATION HEAD
# Purpose: Hierarchical classification head that:
#   SECTION 1: FCSM — Fine-to-Coarse Semantic Mapping (fine→coarse logits)
#   SECTION 2: LOSS COMPUTATION — Fine CE Loss + Coarse CE Loss + TreePathKL Loss
#   Core innovation: Coarse-grained semantic priors constrain fine-grained
#   classification to mitigate class imbalance.
# =====================================================================
class CHLHC(nn.Module):
    def __init__(self, dim, fine2coarse, num_classes=23,head_drop=0.2):

        super(CHLHC, self).__init__()
        self.fine2coarse = fine2coarse
        self.coarse2fine = self._build_coarse2fine()
        self.fine_num = num_classes
        self.coarse_num = self.get_coarse_num()
        self.dim = dim
        self.loss_fine = CrossEntropyLoss()
        self.loss_coarse = CrossEntropyLoss()
        self.lossTK = TreePathKLLoss()

        self.norm_fine = nn.LayerNorm(self.dim, eps=1e-6)
        self.head_fine = nn.Linear(self.dim, self.fine_num)
        self.head_drop_fine = nn.Dropout(head_drop)

        self.g = self._build_transformation_layer(self.fine_num,self.coarse_num)

    def _build_coarse2fine(self):
        coarse2fine = {}
        for fine_idx, coarse_idx in enumerate(self.fine2coarse):
            if coarse_idx not in coarse2fine:
                coarse2fine[coarse_idx] = []
            coarse2fine[coarse_idx].append(fine_idx)
        return coarse2fine
    def get_coarse_num(self):

        unique_values = set(self.fine2coarse)
        unique_count = len(unique_values)
        fine2coarse_total = len(self.fine2coarse)
        assert fine2coarse_total == self.fine_num, \
            f"fine2coarse总数（{fine2coarse_total}）≠ num_classes（{self.fine_num}）"
        return unique_count

    def forward(self, feat_fine,x0,label=None):
        # ===== FCSM: Fine-to-Coarse Semantic Mapping =====
        # Map fine-grained logits to coarse-grained logits via learned transformation
        device = feat_fine.device
        if label is not None:
            coarse_label = torch.tensor(self.fine2coarse).to(device)[label]
            fine_label = label


        feat_fine = self.norm_fine(feat_fine)
        feat_fine = self.head_drop_fine(feat_fine)
        fine_logis = self.head_fine(feat_fine) #B,23

        coarse_logis = self.g(fine_logis)

        # ===== LOSS COMPUTATION: Fine CE + Coarse CE + TreePathKL =====
        if label is not None:
            loss_fine = self.loss_fine(fine_logis, fine_label)
            loss_big = self.loss_coarse(coarse_logis,coarse_label)

            LHV = loss_fine*1.2 + loss_big
            TK = self.lossTK([fine_logis,coarse_logis],[fine_label,coarse_label])
            total = LHV + 0.5*TK

            return fine_logis, total
        else:
            return fine_logis

    def _build_transformation_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Dropout(0.1),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-6),
        )

class TreePathKLLoss(nn.Module):
    def __init__(self, num_levels=2):
        super().__init__()
        self.num_levels = num_levels

    def forward(self, logits_list, labels_list):
        logits_fine, logits_coarse = logits_list
        label_fine, label_coarse = labels_list

        onehot_fine = F.one_hot(label_fine, num_classes=logits_fine.shape[1])
        onehot_coarse = F.one_hot(label_coarse, num_classes=logits_coarse.shape[1])
        gt_concat = torch.cat([onehot_fine, onehot_coarse], dim=1).float()
        gt_dist = gt_concat / gt_concat.sum(dim=1, keepdim=True)

        pred_concat = torch.cat([logits_fine, logits_coarse], dim=1)
        pred_logsoftmax = F.log_softmax(pred_concat, dim=1)

        loss_tk = F.kl_div(pred_logsoftmax, gt_dist, reduction='batchmean')

        return loss_tk







