#!/usr/bin/env python
# =============================================================================
# scripts/backtest.py —— 单独回测入口（加载预训练权重）
# 用法：python scripts/backtest.py --config configs/default.yaml \
#                                   --checkpoint pretrained/itransformer_morl.pth
# =============================================================================

import argparse
import json
import logging
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrl.data.loader import load_real_data
from graphrl.data.dataset import create_enhanced_dataset
from graphrl.models.edygformer import E_DyGFormer
from graphrl.backtest.engine import EnhancedBacktestEngine
from graphrl.rl.policy import PolicyNet, PolicyModelAdapter
from graphrl.utils.seed import set_seed
from graphrl.utils.plotting import plot_enhanced_results


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Graph-RL Quant Backtest')
    parser.add_argument('--config',     type=str, default='configs/default.yaml',
                        help='配置文件路径（YAML）')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径（.pth）')
    parser.add_argument('--split',      type=str, default='test',
                        choices=['val', 'test', 'both'],
                        help='回测数据集划分（默认 test）')
    parser.add_argument('--output_dir', type=str, default='runs/backtest',
                        help='结果输出目录')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[Error] 配置文件不存在: {args.config}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"[Error] 权重文件不存在: {args.checkpoint}")
        sys.exit(1)

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # ------------------------------------------------------------------
    # 1. 数据
    # ------------------------------------------------------------------
    price, bench, panel = load_real_data(
        cfg['data_file_path'],
        max_assets=cfg.get('max_assets', 190),
        max_days=cfg.get('max_days', 2500),
        index_data_path=cfg.get('index_data_path', None),
    )
    dataset, _ = create_enhanced_dataset(
        panel,
        seq_len=cfg.get('seq_len', 60),
        corr_window=cfg.get('corr_window', 10),
        corr_threshold=cfg.get('corr_threshold', 0.3),
    )

    n_all      = len(dataset)
    train_size = int(cfg.get('train_ratio', 0.7) * n_all)
    val_size   = int(cfg.get('val_ratio',   0.2) * n_all)
    val_ds     = dataset[train_size:train_size + val_size]
    test_ds    = dataset[train_size + val_size:]

    # ------------------------------------------------------------------
    # 2. 加载模型
    # ------------------------------------------------------------------
    feat_dim   = dataset[0].x.shape[1]
    num_assets = len(price.columns)

    enc_cfg = dict(cfg['encoder'])
    enc_cfg.update({'node_feat_dim': feat_dim, 'num_assets': num_assets,
                    'seq_len': cfg['seq_len']})
    encoder = E_DyGFormer(**enc_cfg)
    policy  = PolicyNet(encoder, **cfg['policy']).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()
    logger.info(f"已加载权重: {args.checkpoint}")

    adapter_cfg = cfg.get('adapter', {})
    adapter = PolicyModelAdapter(
        policy,
        smooth_weights=adapter_cfg.get('smooth_weights', True),
        smooth_rho=adapter_cfg.get('smooth_rho', 0.3),
    )
    engine_cfg = dict(cfg['engine'])
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # ------------------------------------------------------------------
    # 3. 回测
    # ------------------------------------------------------------------
    if args.split in ('val', 'both'):
        adapter.reset_state()
        bengine   = EnhancedBacktestEngine(**engine_cfg)
        price_val = price.iloc[-(len(val_ds) + len(test_ds)):-len(test_ds)]
        bench_val = bench.iloc[-(len(val_ds) + len(test_ds)):-len(test_ds)]
        res_val   = bengine.run_backtest(adapter, val_ds, price_val, bench_val,
                                         temperature=policy.temperature)
        results['val'] = res_val
        print(f"\nVAL:  ann.ret={res_val['annual_return']:.2%}, "
              f"sharpe={res_val['sharpe_ratio']:.3f}, "
              f"IR={res_val['information_ratio']:.3f}, "
              f"MDD={res_val['max_drawdown']:.2%}")
        try:
            plot_enhanced_results(res_val, bengine.portfolio_value, bengine.benchmark_levels,
                                   save_path=os.path.join(args.output_dir, 'val_nav.png'))
        except Exception as e:
            logger.warning(f"VAL 绘图失败: {e}")

    if args.split in ('test', 'both'):
        adapter.reset_state()
        bengine2   = EnhancedBacktestEngine(**engine_cfg)
        price_test = price.iloc[-len(test_ds):]
        bench_test = bench.iloc[-len(test_ds):]
        res_test   = bengine2.run_backtest(adapter, test_ds, price_test, bench_test,
                                            temperature=policy.temperature)
        results['test'] = res_test
        print(f"TEST: ann.ret={res_test['annual_return']:.2%}, "
              f"sharpe={res_test['sharpe_ratio']:.3f}, "
              f"IR={res_test['information_ratio']:.3f}, "
              f"MDD={res_test['max_drawdown']:.2%}")
        try:
            plot_enhanced_results(res_test, bengine2.portfolio_value, bengine2.benchmark_levels,
                                   save_path=os.path.join(args.output_dir, 'test_nav.png'))
        except Exception as e:
            logger.warning(f"TEST 绘图失败: {e}")

    # ------------------------------------------------------------------
    # 4. 保存结果
    # ------------------------------------------------------------------
    result_path = os.path.join(args.output_dir, 'results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存到 {result_path}")


if __name__ == '__main__':
    main()
