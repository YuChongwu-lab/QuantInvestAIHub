#!/usr/bin/env python
# =============================================================================
# scripts/train.py —— 训练入口
# 用法：python scripts/train.py --config configs/default.yaml
# =============================================================================

# 必须在所有 import 之前设置，避免 Windows 上 OpenMP 冲突
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import sys
import time
import traceback
import warnings

import torch
import yaml

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# 确保项目根目录在 Python 路径中（pip install -e . 后可省略）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrl.curriculum.config import CurriculumConfig
from graphrl.data.loader import load_real_data
from graphrl.data.dataset import create_enhanced_dataset
from graphrl.models.edygformer import E_DyGFormer
from graphrl.backtest.engine import EnhancedBacktestEngine
from graphrl.rl.policy import PolicyNet, PolicyModelAdapter
from graphrl.rl.env import PortfolioEnv
from graphrl.rl.trainer import PPOTrainer, run_curriculum_training
from graphrl.utils.seed import set_seed
from graphrl.utils.plotting import plot_enhanced_results


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_encoder_and_policy(cfg: dict, feat_dim: int, num_assets: int) -> PolicyNet:
    enc_cfg = dict(cfg['encoder'])
    enc_cfg.update({
        'node_feat_dim': feat_dim,
        'num_assets':    num_assets,
        'seq_len':       cfg['seq_len'],
    })
    encoder = E_DyGFormer(**enc_cfg)
    policy  = PolicyNet(encoder, **cfg['policy'])
    return policy


def run_training(cfg: dict):
    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------
    price, bench, panel = load_real_data(
        cfg['data_file_path'],
        max_assets=cfg.get('max_assets', 190),
        max_days=cfg.get('max_days', 2500),
        index_data_path=cfg.get('index_data_path', None),
    )
    dataset, feat_names = create_enhanced_dataset(
        panel,
        seq_len=cfg.get('seq_len', 60),
        corr_window=cfg.get('corr_window', 10),
        corr_threshold=cfg.get('corr_threshold', 0.3),
    )

    n_all      = len(dataset)
    train_size = int(cfg.get('train_ratio', 0.7) * n_all)
    val_size   = int(cfg.get('val_ratio',   0.2) * n_all)
    train_ds   = dataset[:train_size]
    val_ds     = dataset[train_size:train_size + val_size]
    test_ds    = dataset[train_size + val_size:]
    logger.info(f"数据集划分: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ------------------------------------------------------------------
    # 2. 模型构建
    # ------------------------------------------------------------------
    feat_dim   = train_ds[0].x.shape[1]
    num_assets = len(price.columns)
    policy     = build_encoder_and_policy(cfg, feat_dim, num_assets).to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # ------------------------------------------------------------------
    # 3. 训练
    # ------------------------------------------------------------------
    engine_cfg = dict(cfg['engine'])
    reward_cfg = dict(cfg['reward'])
    morl_cfg   = cfg.get('morl', {})

    ppo_kwargs = dict(cfg['ppo'])
    ppo_kwargs.update({
        'morl_enable':       morl_cfg.get('enabled', False),
        'morl_aux_weight':   morl_cfg.get('aux_weight', 0.5),
        'morl_alpha':        morl_cfg.get('alpha', 1.0),
        'morl_beta':         morl_cfg.get('beta', 0.5),
        'use_grad_surgery':  morl_cfg.get('use_grad_surgery', True),
        'adaptive_weights':  morl_cfg.get('adaptive_weights', True),
    })

    curriculum_cfg = cfg.get('curriculum', {})
    if curriculum_cfg.get('enabled', False):
        # 课程学习模式
        config_file = curriculum_cfg.get('config_file')
        if config_file and os.path.exists(config_file):
            curriculum = CurriculumConfig.from_yaml(config_file)
            logger.info(f"加载课程配置: {config_file}")
        else:
            curriculum = CurriculumConfig.create_default()
            logger.info("使用默认3阶段课程配置")

        run_curriculum_training(
            policy=policy,
            price_df=price,
            train_ds=train_ds,
            base_engine_cfg=engine_cfg,
            base_reward_cfg=reward_cfg,
            trainer_kwargs=ppo_kwargs,
            curriculum_config=curriculum,
            horizon=cfg.get('horizon', 128),
        )
    else:
        # 普通 PPO 模式
        env     = PortfolioEnv(policy, train_ds, price, engine_cfg, reward_cfg)
        trainer = PPOTrainer(policy, env, **ppo_kwargs, device=device)
        epochs  = cfg.get('epochs', 30)
        for ep in range(1, epochs + 1):
            buf = trainer.collect(horizon=cfg.get('horizon', 128), stochastic=True)
            trainer.update(buf)
            if ep % 5 == 0 or ep == epochs:
                last_V = float(buf[-1].V.detach().cpu().item())
                logger.info(f"[PPO] epoch {ep}/{epochs} | V≈{last_V:,.2f}")

    # ------------------------------------------------------------------
    # 4. 回测
    # ------------------------------------------------------------------
    adapter_cfg = cfg.get('adapter', {})
    adapter = PolicyModelAdapter(
        policy,
        smooth_weights=adapter_cfg.get('smooth_weights', True),
        smooth_rho=adapter_cfg.get('smooth_rho', 0.3),
    )

    # 验证集
    adapter.reset_state()
    bengine   = EnhancedBacktestEngine(**engine_cfg)
    price_val = price.iloc[-(len(val_ds) + len(test_ds)):-len(test_ds)]
    bench_val = bench.iloc[-(len(val_ds) + len(test_ds)):-len(test_ds)]
    res_val   = bengine.run_backtest(adapter, val_ds, price_val, bench_val,
                                     temperature=policy.temperature)

    # 测试集
    adapter.reset_state()
    bengine2    = EnhancedBacktestEngine(**engine_cfg)
    price_test  = price.iloc[-len(test_ds):]
    bench_test  = bench.iloc[-len(test_ds):]
    res_test    = bengine2.run_backtest(adapter, test_ds, price_test, bench_test,
                                        temperature=policy.temperature)

    # ------------------------------------------------------------------
    # 5. 结果输出
    # ------------------------------------------------------------------
    print("\n=== Graph-RL Evaluation ===")
    print(f"VAL:  ann.ret={res_val['annual_return']:.2%}, "
          f"sharpe={res_val['sharpe_ratio']:.3f}, "
          f"IR={res_val['information_ratio']:.3f}, "
          f"MDD={res_val['max_drawdown']:.2%}")
    print(f"TEST: ann.ret={res_test['annual_return']:.2%}, "
          f"sharpe={res_test['sharpe_ratio']:.3f}, "
          f"IR={res_test['information_ratio']:.3f}, "
          f"MDD={res_test['max_drawdown']:.2%}")

    # 保存结果 JSON
    output_cfg  = cfg.get('output', {})
    result_path = output_cfg.get('result_json', 'runs/results.json')
    os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({'val': res_val, 'test': res_test}, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存到 {result_path}")

    # 保存模型权重
    ckpt_path = os.path.join(os.path.dirname(result_path), 'model.pth')
    torch.save(policy.state_dict(), ckpt_path)
    logger.info(f"模型权重已保存到 {ckpt_path}")

    # 可视化
    try:
        plot_enhanced_results(res_val, bengine.portfolio_value, bengine.benchmark_levels,
                               save_path=output_cfg.get('plot_val', 'runs/val_nav.png'))
        plot_enhanced_results(res_test, bengine2.portfolio_value, bengine2.benchmark_levels,
                               save_path=output_cfg.get('plot_test', 'runs/test_nav.png'))
    except Exception as e:
        logger.warning(f"绘图失败: {e}")

    return policy, res_val, res_test


def main():
    parser = argparse.ArgumentParser(description='Graph-RL Quant Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径（YAML）')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[Error] 配置文件不存在: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)
    logger.info(f"加载配置: {args.config}")

    t0 = time.time()
    try:
        run_training(cfg)
        logger.info(f"训练完成，总耗时: {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
