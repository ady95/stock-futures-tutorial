"""07-5 실습: 보상 설계별 PPO 비교 + 위험관리 엔진 효과.

실행:  python ch07/compare_rewards.py
1) 6장 기본과 7장 실험 5개의 평가 구간 성과를 시드 3개 중앙값·범위로 한 표에
2) R2 모델이 전체 청산(행동 6)을 고른 비율, 행동 마스크가 선택지를 줄인 날의 비율
3) 위험관리 엔진을 기준선 전략(B1n, B2)에 붙였을 때의 변화 (합성 평가 구간, 실제 2015~2025)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.backtest import run_many  # noqa: E402
from hedgelab.episode import build_all  # noqa: E402
from hedgelab.metrics import summary  # noqa: E402
from hedgelab.risk import RiskEngine, with_risk  # noqa: E402
from hedgelab.rl import evaluate, load_config, load_episodes, make_env, model_class, predict  # noqa: E402
from hedgelab.strategies import BASELINES  # noqa: E402

EXPERIMENTS = [("ppo_r1_synthetic", "기본 (페널티 없음)"), ("ch07_dd05", "낙폭 λ1=0.5"),
               ("ch07_dd20", "낙폭 λ1=2.0"), ("ch07_dd_margin", "낙폭+증거금"),
               ("ch07_all", "낙폭+증거금+거래"), ("ch07_r2_risk", "R2+위험엔진+마스킹")]
KEYS = ["순손익(원)", "MDD(원)", "CVaR5%(원)", "평균 노출(정규 계약)", "거래 계약 수"]

# 1) 보상 설계별 비교
rows = []
for name, label in EXPERIMENTS:
    cfg = load_config(f"configs/{name}.yaml")
    eps = load_episodes(cfg, "test")
    res = [summary(evaluate(model_class(cfg).load(f"models/{name}_seed{s}.zip", device="cpu"), eps, cfg))
           for s in cfg["train"]["seeds"]]
    row = {"실험": label}
    for k in KEYS:
        v = [r[k] for r in res]
        row[k] = f"{np.median(v):,.0f} ({min(v):,.0f} ~ {max(v):,.0f})" if "원" in k or "수" in k \
            else f"{np.median(v):.2f} ({min(v):.2f} ~ {max(v):.2f})"
    rows.append(row)
print("== 평가 구간 성과: 중앙값 (최소 ~ 최대), 시드 3개 ==")
print(pd.DataFrame(rows).set_index("실험").to_string())

# 2) R2: 전체 청산 비율과 마스크
cfg = load_config("configs/ch07_r2_risk.yaml")
eps = load_episodes(cfg, "test")
print("\n== R2 모델의 행동 ==")
for s in cfg["train"]["seeds"]:
    model = model_class(cfg).load(f"models/ch07_r2_risk_seed{s}.zip", device="cpu")
    env = make_env(eps, cfg, random_episode=False)
    days = flat = masked = 0
    for i in range(len(eps)):
        obs, _ = env.reset(options={"episode_index": i})
        done = False
        while not done:
            m = env.action_masks()
            masked += int(not m.all())
            a = predict(model, env, obs)
            flat += int(a == 6)
            days += 1
            obs, _, done, _, _ = env.step(a)
    print(f"  seed{s}: 결정 {days}회, 전체 청산 선택 {flat}회({flat / days:.0%}), "
          f"마스크로 선택지가 줄어든 날 {masked}회({masked / days:.0%})")

# 3) 위험관리 엔진을 기준선에 붙이기
print("\n== 위험관리 엔진 효과 (손실 한도 6%, 증거금 한도 40%) ==")
data = {"합성 평가 구간": build_all("data/synthetic", 2024, 2025, synthetic=True),
        "실제 2015~2025": build_all("data/real", 2015, 2025)}
for dname, episodes in data.items():
    out = {}
    for bname in ("B1n 고정 헤지(롤오버 없음)", "B2 ATR 규칙형"):
        base = BASELINES[bname]
        out[bname] = summary(run_many(episodes, base))
        engine = RiskEngine(max_margin_use=0.40, stop_loss=0.06, flatten_on_stop=False)
        out[bname + " + 엔진"] = summary(run_many(episodes, with_risk(base, engine)))
        out[bname + " + 엔진"]["엔진 개입"] = len(engine.log)
    table = pd.DataFrame(out).reindex(["순손익(원)", "MDD(원)", "CVaR5%(원)", "평균 노출(정규 계약)", "엔진 개입"])
    print(f"[{dname}]")
    with pd.option_context("display.width", 250):
        print(table.to_string())
