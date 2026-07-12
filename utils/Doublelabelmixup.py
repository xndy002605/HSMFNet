import torch
import numpy as np
from timm.data.mixup import one_hot


class DoubleLabelMixup:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1,
                 num_classes_small=42, num_classes_big=10, prob=1.0, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.num_classes_small = num_classes_small  # 小分类数
        self.num_classes_big = num_classes_big  # 大分类数
        self.prob = prob  # 应用mixup/cutmix的概率
        self.switch_prob = switch_prob  # mixup和cutmix的切换概率

        self.mixup_enabled = mixup_alpha > 0
        self.cutmix_enabled = cutmix_alpha > 0

    def _params(self):
        """生成mixup/cutmix的混合参数"""
        if self.mixup_enabled and self.cutmix_enabled:
            use_cutmix = np.random.rand() < self.switch_prob
            alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        elif self.cutmix_enabled:
            use_cutmix = True
            alpha = self.cutmix_alpha
        else:
            use_cutmix = False
            alpha = self.mixup_alpha

        lam = 1.0
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)  # 生成混合比例
        return lam, use_cutmix

    def _cutmix_bbox_and_lam(self, img_shape, lam):
        """生成cutmix的边界框和调整后的混合比例"""
        H, W = img_shape[-2], img_shape[-1]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机生成边界框中心
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 计算边界框坐标（确保在图像范围内）
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 调整混合比例（实际裁剪面积占比）
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        return (bbx1, bby1, bbx2, bby2), lam

    def __call__(self, x, target):
        """
        输入：
            x: 图像张量 (batch_size, C, H, W)
            target: 双标签张量 (batch_size, 2)，第一列是小类标签，第二列是大类标签
        输出：
            x_mixed: 混合后的图像
            target_mixed: 混合后的双标签 (small_mixed, big_mixed)，每个为独热编码的软标签
        """
        # 解析双标签（从二维张量中提取，确保形状为(batch_size,)）
        # 假设target形状为(batch_size, 2)，[:,0]为小类标签，[:,1]为大类标签


        small_labels = target[:, 0]  # 提取小类标签并转为长整数类型
        big_labels = target[:, 1]   # 提取大类标签并转为长整数类型
        batch_size = x.size(0)

        # 验证标签形状
        if small_labels.shape != (batch_size,):
            raise ValueError(f"small_labels 形状错误，应为 (batch_size,)，实际为 {small_labels.shape}")
        if big_labels.shape != (batch_size,):
            raise ValueError(f"big_labels 形状错误，应为 (batch_size,)，实际为 {big_labels.shape}")

        device = x.device

        # 以概率self.prob决定是否应用mixup/cutmix
        if np.random.rand() < self.prob:
            lam, use_cutmix = self._params()
            # 生成打乱的索引（确保与batch_size一致）
            shuffled_idx = torch.randperm(batch_size, device=device)

            # 图像混合（mixup或cutmix）
            if use_cutmix:
                # CutMix逻辑：裁剪并替换区域
                bbox, lam = self._cutmix_bbox_and_lam(x.shape, lam)
                bbx1, bby1, bbx2, bby2 = bbox
                x[:, :, bby1:bby2, bbx1:bbx2] = x[shuffled_idx, :, bby1:bby2, bbx1:bbx2]
            else:
                # Mixup逻辑：线性混合
                x = lam * x + (1 - lam) * x[shuffled_idx]

            # 标签混合：对小分类和大分类分别生成独热编码并混合
            # 小分类标签混合
            small_one_hot = one_hot(
                small_labels,
                num_classes=self.num_classes_small,
            )
            small_shuffled_one_hot = one_hot(
                small_labels[shuffled_idx],  # 使用打乱的索引
                num_classes=self.num_classes_small,

            )
            small_mixed = lam * small_one_hot + (1 - lam) * small_shuffled_one_hot

            # 大分类标签混合
            big_one_hot = one_hot(
                big_labels,
                num_classes=self.num_classes_big,

            )
            big_shuffled_one_hot = one_hot(
                big_labels[shuffled_idx],  # 使用打乱的索引
                num_classes=self.num_classes_big,

            )
            big_mixed = lam * big_one_hot + (1 - lam) * big_shuffled_one_hot


            return x, (small_mixed, big_mixed)

        # 不应用混合时，直接返回原始图像和独热编码标签
        else:
            small_one_hot = one_hot(
                small_labels,
                num_classes=self.num_classes_small,

            )
            big_one_hot = one_hot(
                big_labels,
                num_classes=self.num_classes_big,

            )
            return x, (small_one_hot, big_one_hot)