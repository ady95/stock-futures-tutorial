"""04-5 실습: 기준선 전략 백테스트 (B0·B1·B1n·B2).

실행:  python ch04/run_baselines.py synthetic            합성 데이터 (seed=4, 2020~2025)
       python ch04/run_baselines.py real [데이터폴더]     실제 데이터 (기본 data/real, 2015~2025)
결과:  표 출력 + results/ch04_<데이터>_equity.png
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab import plotting  # noqa: E402
from hedgelab.backtest import run_many  # noqa: E402
from hedgelab.episode import build_all  # noqa: E402
from hedgelab.metrics import summary  # noqa: E402
from hedgelab.strategies import BASELINES  # noqa: E402

plt = plotting.setup()

kind = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
if kind == "synthetic":
    episodes = build_all("data/synthetic", 2020, 2025, synthetic=True)
else:
    episodes = build_all(sys.argv[2] if len(sys.argv) > 2 else "data/real", 2015, 2025)
print(f"데이터: {kind}, 에피소드 {len(episodes)}개, "
      f"{episodes[0].index[0]:%Y-%m-%d} ~ {episodes[-1].index[-1]:%Y-%m-%d}")

EQUITY0 = 50_000_000
rows, curves = {}, {}
for name, policy in BASELINES.items():
    rec = run_many(episodes, policy, equity0=EQUITY0)
    rows[name] = summary(rec, EQUITY0)
    curves[name] = rec.equity

table = pd.DataFrame(rows)
with pd.option_context("display.width", 200, "display.max_columns", 10):
    print(table.to_string())

os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
for name, eq in curves.items():
    ax.plot(range(len(eq)), (eq - EQUITY0) / 1e6, label=name, lw=1.2)
ax.set_xlabel("거래일 (에피소드를 이어 붙인 순서)")
ax.set_ylabel("누적 손익 (백만 원)")
ax.set_title(f"기준선 전략 누적 손익 — {kind}")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
path = f"results/ch04_{kind}_equity.png"
fig.savefig(path, dpi=120)
print(f"그래프 저장: {path}")
