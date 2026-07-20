# =====================================================================
# CHLHC: Coarse-Fine Linked Supervision Hierarchical Classification Head
# =====================================================================
# Core Innovation: Hierarchical classification with tree-structured supervision.
# Coarse-grained semantic priors constrain fine-grained classification to
# mitigate class imbalance. Coarse logits are derived from fine logits via
# a learnable transformation, ensuring tree-structured consistency.
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CHLHC(nn.Module):
    def __init__(self, dim, fine2coarse_mapping, num_classes, **kwargs):
        super().__init__()
        self.mapping = fine2coarse_mapping
        self.num_fine = num_classes
        self.num_coarse = len(set(fine2coarse_mapping))
        self.coarse2fine = self._build_hierarchy_index()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, self.num_fine)
        self.dropout = nn.Dropout(kwargs.get('drop_rate', 0.2))

        self.bridge = nn.Sequential(
            nn.LayerNorm(self.num_fine, eps=1e-6),
            nn.Dropout(0.1),
            nn.Linear(self.num_fine, self.num_coarse),
            nn.GELU(),
            nn.LayerNorm(self.num_coarse, eps=1e-6)
        )

        self.fine_loss = nn.CrossEntropyLoss()
        self.coarse_loss = nn.CrossEntropyLoss()
        self.tree_loss = _TreePathKLLoss()

    def _build_hierarchy_index(self):
        index = {}
        for f_idx, c_idx in enumerate(self.mapping):
            index.setdefault(c_idx, []).append(f_idx)
        return index

    def forward(self, features, labels=None):
        x = self.dropout(self.norm(features))
        fine_logits = self.head(x)
        coarse_logits = self.bridge(fine_logits)

        if labels is not None:
            coarse_labels = torch.tensor(self.mapping).to(labels.device)[labels]
            L_fine = self.fine_loss(fine_logits, labels)
            L_coarse = self.coarse_loss(coarse_logits, coarse_labels)
            L_tree = self.tree_loss([fine_logits, coarse_logits], [labels, coarse_labels])
            return fine_logits, L_fine * 1.2 + L_coarse + 0.5 * L_tree
        return fine_logits


class _TreePathKLLoss(nn.Module):
    """Tree-structured KL divergence enforcing hierarchical consistency."""

    def forward(self, logits_list, labels_list):
        fine_logits, coarse_logits = logits_list
        fine_labels, coarse_labels = labels_list

        gt = torch.cat([
            F.one_hot(fine_labels, fine_logits.shape[1]),
            F.one_hot(coarse_labels, coarse_logits.shape[1])
        ], dim=1).float()

        pred = torch.cat([fine_logits, coarse_logits], dim=1)
        gt_dist = gt / gt.sum(dim=1, keepdim=True)
        return F.kl_div(F.log_softmax(pred, dim=1), gt_dist, reduction='batchmean')
