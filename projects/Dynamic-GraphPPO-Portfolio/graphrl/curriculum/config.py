# =============================================================================
# core/curriculum.py —— 课程学习配置类
# 包含：StageConfig、CurriculumConfig
# =============================================================================

import json
import logging
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """单个训练阶段的配置"""
    name: str
    epochs: int
    max_pos: float          # 单资产最大持仓比例
    cost: float             # 交易成本率
    lambda_turnover: float  # 换手惩罚系数
    loss_weight: Dict[str, float] = field(
        default_factory=lambda: {'mse': 1.0, 'ic': 0.0, 'return': 0.0}
    )

    def __post_init__(self):
        assert self.epochs > 0, f"epochs必须>0，当前: {self.epochs}"
        assert 0 < self.max_pos <= 1, f"max_pos必须在(0,1]，当前: {self.max_pos}"
        assert self.cost >= 0, f"cost必须>=0，当前: {self.cost}"
        assert self.lambda_turnover >= 0, f"lambda_turnover必须>=0，当前: {self.lambda_turnover}"
        required_keys = {'mse', 'ic', 'return'}
        assert set(self.loss_weight.keys()) == required_keys, \
            f"loss_weight必须包含{required_keys}，当前: {set(self.loss_weight.keys())}"


@dataclass
class CurriculumConfig:
    """完整的课程学习配置"""
    stages: List[StageConfig]

    @classmethod
    def from_dict(cls, config: Dict) -> 'CurriculumConfig':
        stages = [StageConfig(**stage) for stage in config['stages']]
        return cls(stages=stages)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CurriculumConfig':
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @classmethod
    def from_json(cls, json_path: str) -> 'CurriculumConfig':
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls.from_dict(config)

    def to_yaml(self, yaml_path: str):
        config = {'stages': [asdict(stage) for stage in self.stages]}
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    def to_json(self, json_path: str):
        config = {'stages': [asdict(stage) for stage in self.stages]}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def create_default(cls) -> 'CurriculumConfig':
        """创建默认3阶段配置"""
        stages = [
            StageConfig(
                name='Stage-1-Warmup', epochs=8, max_pos=0.30, cost=0.001,
                lambda_turnover=0.02,
                loss_weight={'mse': 1.0, 'ic': 0.0, 'return': 0.0}
            ),
            StageConfig(
                name='Stage-2-Balance', epochs=10, max_pos=0.25, cost=0.002,
                lambda_turnover=0.04,
                loss_weight={'mse': 0.5, 'ic': 0.5, 'return': 0.0}
            ),
            StageConfig(
                name='Stage-3-Finetune', epochs=12, max_pos=0.20, cost=0.003,
                lambda_turnover=0.05,
                loss_weight={'mse': 0.0, 'ic': 0.2, 'return': 1.0}
            ),
        ]
        return cls(stages=stages)

    @classmethod
    def create_progressive(cls,
                           num_stages: int = 5,
                           total_epochs: int = 40,
                           start_max_pos: float = 0.35,
                           end_max_pos: float = 0.15,
                           start_cost: float = 0.0005,
                           end_cost: float = 0.004,
                           start_lambda: float = 0.01,
                           end_lambda: float = 0.08) -> 'CurriculumConfig':
        """
        自动生成渐进式课程

        参数：
            num_stages: 阶段数
            total_epochs: 总训练轮数
            start_max_pos: 起始最大持仓比例（宽松）
            end_max_pos: 最终最大持仓比例（严格）
            start_cost: 起始交易成本（低）
            end_cost: 最终交易成本（高）
            start_lambda: 起始换手惩罚（低）
            end_lambda: 最终换手惩罚（高）
        """
        stages = []
        epochs_per_stage = max(1, total_epochs // num_stages)

        for i in range(num_stages):
            progress = i / max(1, num_stages - 1)

            max_pos  = start_max_pos  + (end_max_pos  - start_max_pos)  * progress
            cost     = start_cost     + (end_cost     - start_cost)     * progress
            lambda_t = start_lambda   + (end_lambda   - start_lambda)   * progress

            if i < num_stages // 3:
                loss_weight = {'mse': 1.0, 'ic': 0.0, 'return': 0.0}
            elif i < 2 * num_stages // 3:
                loss_weight = {'mse': 0.5, 'ic': 0.5, 'return': 0.0}
            else:
                ic_weight = 0.2 + 0.3 * (progress - 2 / 3) * 3
                loss_weight = {'mse': 0.0, 'ic': ic_weight, 'return': 1.0 - ic_weight}

            epochs = (
                epochs_per_stage if i < num_stages - 1
                else (total_epochs - epochs_per_stage * (num_stages - 1))
            )
            stages.append(StageConfig(
                name=f'Stage-{i + 1}-Progressive',
                epochs=epochs,
                max_pos=round(max_pos, 3),
                cost=round(cost, 5),
                lambda_turnover=round(lambda_t, 3),
                loss_weight=loss_weight
            ))

        return cls(stages=stages)

    def validate_progression(self) -> bool:
        """验证课程是否是合理的渐进式（难度递增）"""
        for i in range(len(self.stages) - 1):
            curr = self.stages[i]
            next_stage = self.stages[i + 1]
            if next_stage.max_pos > curr.max_pos:
                logger.warning(f"{curr.name} -> {next_stage.name}: max_pos变宽松（可能不合理）")
            if next_stage.cost < curr.cost:
                logger.warning(f"{curr.name} -> {next_stage.name}: cost降低（可能不合理）")
            if next_stage.lambda_turnover < curr.lambda_turnover:
                logger.warning(f"{curr.name} -> {next_stage.name}: lambda_turnover降低（可能不合理）")
        return True

    def print_summary(self):
        """打印课程摘要"""
        print("\n" + "=" * 60)
        print("课程学习配置摘要".center(60))
        print("=" * 60)
        print(f"总阶段数: {len(self.stages)}")
        print(f"总训练轮数: {sum(s.epochs for s in self.stages)}")
        print("-" * 60)
        for i, stage in enumerate(self.stages, 1):
            print(f"\n【{stage.name}】")
            print(f"  轮数: {stage.epochs}")
            print(f"  约束: max_pos={stage.max_pos:.2%}, cost={stage.cost:.4%}, λ_turn={stage.lambda_turnover:.3f}")
            print(
                f"  损失权重: MSE={stage.loss_weight['mse']:.1f}, "
                f"IC={stage.loss_weight['ic']:.1f}, "
                f"Return={stage.loss_weight['return']:.1f}"
            )
        print("=" * 60 + "\n")
