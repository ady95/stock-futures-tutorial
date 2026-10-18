"""04-1 회계 검증: 손으로 계산할 수 있는 작은 에피소드로 엔진을 검사한다.

실행:  python -m pytest -q tests
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.backtest import CostModel, run_episode  # noqa: E402
from hedgelab.strategies import b1_fixed, b1n_fixed_no_roll  # noqa: E402

NO_COST = CostModel(fee_rate=0.0, slip_ticks=0.0)


def toy_episode():
    """5일짜리 에피소드. 3일째(인덱스 2) 마감에 미니 a 만기, 이후 b로 넘어간다."""
    days = pd.bdate_range("2025-11-10", periods=5)
    ep = pd.DataFrame({
        "reg":    [500.0, 510.0, 505.0, 495.0, 490.0],
        "reg_high": [501.0, 511.0, 506.0, 496.0, 491.0],
        "reg_low":  [499.0, 509.0, 504.0, 494.0, 489.0],
        "mini_a": [499.6, 509.4, 504.6, None, None],
        "mini_b": [500.2, 510.2, 505.2, 495.1, 490.0],
        "near":   ["a", "a", "a", "b", "b"],
        "a_last": [False, False, True, False, False],
    }, index=days)
    ep.attrs.update(year=2025, month=12)
    return ep


def test_full_hedge_pnl_equals_spread_change():
    rec = run_episode(toy_episode(), b1_fixed, equity0=0, costs=NO_COST)
    # 1일째: 정규 +10 → +2,500,000 / 미니 a +9.8 × 5계약 매도 → −2,450,000
    assert rec.pnl_reg.iloc[1] == pytest.approx(2_500_000)
    assert rec.pnl_mini.iloc[1] == pytest.approx(-2_450_000)
    # 3일째(b로 롤오버 후): 정규 −10, 미니 b −10.1 → 2,500,000 × (−10 + 10.1) = +25,000
    assert rec.pnl_reg.iloc[3] + rec.pnl_mini.iloc[3] == pytest.approx(25_000)


def test_no_roll_leaves_naked_long_after_expiry():
    rec = run_episode(toy_episode(), b1n_fixed_no_roll, equity0=0, costs=NO_COST)
    assert list(rec.h) == [5, 5, 0, 0, 0]
    # 미니 만기 후 정규 1계약만: 3일째 −10pt → −2,500,000, 4일째 −5pt → −1,250,000
    assert rec.pnl_mini.iloc[3] == 0
    assert rec.pnl_reg.iloc[3] == pytest.approx(-2_500_000)
    assert rec.pnl_reg.iloc[4] == pytest.approx(-1_250_000)


def test_costs_charged_on_trades_only():
    costs = CostModel(fee_rate=0.0, slip_ticks=1.0)
    rec = run_episode(toy_episode(), b1_fixed, equity0=0, costs=costs)
    # 첫날: 정규 1계약(0.05pt × 25만 = 12,500) + 미니 5계약(0.02pt × 5만 × 5 = 5,000)
    assert rec.cost.iloc[0] == pytest.approx(17_500)
    # a 최종결제는 비용 없음, b 5계약 새로 매도 → 5,000
    assert rec.cost.iloc[2] == pytest.approx(5_000)
    # 마지막 날 만기 결제는 비용 없음, 중간 날은 거래 없음
    assert rec.cost.iloc[1] == 0 and rec.cost.iloc[3] == 0 and rec.cost.iloc[4] == 0


def test_pnl_attribution_adds_up():
    for policy in (b1_fixed, b1n_fixed_no_roll):
        rec = run_episode(toy_episode(), policy, equity0=0, costs=NO_COST)
        total = rec.pnl_reg + rec.pnl_mini
        assert (rec.pnl_dir + rec.pnl_spread - total).abs().max() < 1e-6   # 부동소수점 오차 허용


def test_last_day_flat():
    rec = run_episode(toy_episode(), b1_fixed, equity0=0, costs=NO_COST)
    assert rec.n_reg.iloc[-1] == 0 and rec.h.iloc[-1] == 0
