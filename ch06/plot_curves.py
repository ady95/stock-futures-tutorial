"""06-4 실습: 학습 곡선 — 학습 구간과 평가 구간의 보상 합을 함께 그린다.

실행:  python ch06/plot_curves.py ppo_r1_synthetic [다른 실험 이름...]
입력:  results/<이름>_seed<N>_curve.csv  (train_ppo.py 가 저장)
결과:  results/ch06_curves_<이름>.png
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab import plotting  # noqa: E402
from hedgelab.backtest import run_episode  # noqa: E402
from hedgelab.rl import load_config, load_episodes  # noqa: E402
from hedgelab.strategies import b1_fixed, make_b2_atr  # noqa: E402


def baseline_return(episodes, policy, equity0=50_000_000):
    """기준선 전략의 '에피소드 보상 합 평균' — 100 × log(끝 순자산 / 시작 순자산)의 평균."""
    return float(np.mean([100 * np.log(run_episode(ep, policy, equity0).equity.iloc[-1] / equity0)
                          for ep in episodes]))


plt = plotting.setup()
names = sys.argv[1:] or ["ppo_r1_synthetic"]
for name in names:
    cfg = load_config(f"configs/{name}.yaml")
    base = {split: {"B1": baseline_return(load_episodes(cfg, split), b1_fixed),
                    "B2": baseline_return(load_episodes(cfg, split), make_b2_atr())}
            for split in ("train", "test")}
    print(f"기준선 보상 합 평균: {base}")
    files = sorted(glob.glob(f"results/{name}_seed*_curve.csv"))
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)   # 세로 배치: 책(A4) 폭에서도 글자가 읽히게
    for f in files:
        df = pd.read_csv(f)
        seed = f.split("_seed")[1].split("_")[0]
        axes[0].plot(df.steps / 1000, df.train, lw=1.3, label=f"seed {seed}")
        axes[1].plot(df.steps / 1000, df.test, lw=1.3, label=f"seed {seed}")
        print(f"{name} seed{seed}: 최고 평가 {df.test.max():.3f} ({int(df.steps[df.test.idxmax()])}스텝), "
              f"마지막 학습 {df.train.iloc[-1]:.3f} / 평가 {df.test.iloc[-1]:.3f}")
    for ax, split, title in zip(axes, ("train", "test"), ["학습 구간 (2020~2023)", "평가 구간 (2024~2025)"]):
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(base[split]["B1"], color="black", ls=":", lw=1.2, label="B1 고정 헤지")
        ax.axhline(base[split]["B2"], color="black", ls="--", lw=1.2, label="B2 ATR 규칙형")
        ax.set_title(title)
        ax.set_ylabel("에피소드 보상 합 평균")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("학습 스텝 (천)")
    axes[1].legend(ncol=2)
    fig.suptitle(f"학습 곡선 — {name}")
    fig.tight_layout()
    out = f"results/ch06_curves_{name}.png"
    fig.savefig(out, dpi=120)
    print(f"그래프 저장: {out}")
