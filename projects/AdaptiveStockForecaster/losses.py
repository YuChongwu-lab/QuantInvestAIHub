
import torch
import torch.nn as nn

class DirectionalLoss(nn.Module):
    """
    方向损失：奖励正确的涨跌方向预测
    原理：
    - 如果预测方向正确，损失较小
    - 如果预测方向错误，损失较大
    - 结合了分类和回归的优点
    """

    def __init__(self, alpha=1.0, beta=2.0, use_soft=True):
        """
        Args:
            alpha: 方向正确时的权重
            beta: 方向错误时的权重（通常 beta > alpha）
            use_soft: 是否使用软方向匹配
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_soft = use_soft

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T) 预测的收益率
            target: (B, T) 真实的收益率
        Returns:
            loss: scalar
        """
        if self.use_soft:
            # pred * target > 0 表示方向一致
            direction_agreement = pred * target
            # sigmoid 将其映射到 [0, 1]
            direction_correct = torch.sigmoid(direction_agreement * 10)
        else:
            # 原始硬匹配
            direction_correct = (torch.sign(pred) == torch.sign(target)).float()

        # MSE 基础损失
        mse = (pred - target) ** 2

        # 方向正确时降低权重，方向错误时增加权重
        # direction_correct 接近 1 → 权重接近 alpha
        # direction_correct 接近 0 → 权重接近 beta
        weights = self.alpha + (self.beta - self.alpha) * (1 - direction_correct)

        # 加权损失
        loss = (weights * mse).mean()

        return loss


