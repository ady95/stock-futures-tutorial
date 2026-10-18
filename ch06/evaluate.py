"""06-5 실습: 학습한 PPO를 평가 구간에서 기준선과 비교하고, 고른 헤지비율을 분석한다.

실행:  python ch06/evaluate.py configs/ppo_r1_synthetic.yaml [split]     split 기본값 test
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.backtest import run_many  # noqa: E402
from hedgelab.metrics import summary  # noqa: E402
from hedgelab.rl import action_profile, evaluate, load_config, load_episodes, model_class  # noqa: E402
from hedgelab.strategies import BASELINES  # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/ppo_r1_synthetic.yaml")
split = sys.argv[2] if len(sys.argv) > 2 else "test"
episodes = load_episodes(cfg, split)
print(f"[{cfg['name']}] {split} 에피소드 {len(episodes)}개 "
      f"{episodes[0].index[0]:%Y-%m-%d} ~ {episodes[-1].index[-1]:%Y-%m-%d}")

KEYS = ["순손익(원)", "MDD(원)", "Sharpe", "CVaR5%(원)", "평균 노출(정규 계약)", "거래 계약 수",
        "방향 손익(원)", "스프레드 손익(원)", "비용(원)"]
rows = {name: summary(run_many(episodes, policy)) for name, policy in BASELINES.items()}
profiles = {}
for seed in cfg["train"]["seeds"]:
    path = f"models/{cfg['name']}_seed{seed}.zip"
    if not os.path.exists(path):
        continue
    rec = evaluate(model_class(cfg).load(path, device="cpu"), episodes, cfg)
    rows[f"PPO seed{seed}"] = summary(rec)
    rows[f"PPO seed{seed}"]["위험엔진 개입"] = len(rec.attrs["risk_log"])
    profiles[seed] = action_profile(rec)

if cfg.get("risk"):
    KEYS.append("위험엔진 개입")
table = pd.DataFrame(rows).reindex(KEYS)
with pd.option_context("display.width", 250, "display.max_columns", 12):
    print(table.to_string())

print("\nPPO가 고른 미니 매도 계약 수 분포 (비율)")
for seed, (dist, by_phase) in profiles.items():
    d = " ".join(f"h{int(k)}:{v:.2f}" for k, v in dist.items())
    print(f"  seed{seed}: {d}  | 평균 h — 미니 a 구간 {by_phase.get('a', float('nan')):.2f}, "
          f"만기 후(b) 구간 {by_phase.get('b', float('nan')):.2f}")
