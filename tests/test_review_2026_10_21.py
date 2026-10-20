"""2026-10-21 검수에서 재현된 문제의 회귀 테스트.

P1-01 운영 중 누락일을 모델로 백필하던 문제 → 직전 포지션 유지
P1-02 손실 한도 처리 뒤 증거금 검사를 건너뛰던 문제 → 두 한도 모두 적용
P2-01 시세 검사 대상 → 정규 항상, 미니 a 는 만기일까지, 미니 b 는 a 만기일부터
P2-02 최신 결정일 일봉이 없을 때 KeyError → LedgerDataMissing

실행:  python -m pytest -q tests
"""
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedgelab.backtest import MarginModel  # noqa: E402
from hedgelab.live import LedgerDataMissing, PaperBook, ledger_from_decisions, quote_problem, run_once  # noqa: E402
from hedgelab.risk import RiskEngine  # noqa: E402

MARGIN = MarginModel()


# ---------------------------------------------------------------- P1-02
def test_loss_limit_then_margin_limit_both_apply():
    """검수 재현 입력: R1, 시작 5,000만, 순자산 100만, 가격 500, 목표 (1, 0), 증거금 한도 40%."""
    eng = RiskEngine(max_margin_use=0.40, stop_loss=0.06, flatten_on_stop=False)
    adj = eng.adjust(1, 0, equity=1_000_000, start_equity=50_000_000, equity0=50_000_000,
                     reg_price=500.0, margin=MARGIN)
    assert adj == (0, 0)                                     # 완전 헤지(250만 원)도 한도(40만 원) 초과 → 청산
    assert eng.log[-1]["reason"] == "손실 한도 + 증거금 한도"


@pytest.mark.parametrize("equity", [-5_000_000, 0, 1_000_000, 3_000_000, 10_000_000, 50_000_000])
@pytest.mark.parametrize("flatten", [False, True])
def test_returned_position_never_exceeds_margin(equity, flatten):
    """불변조건: 어떤 목표·순자산에서도 반환 포지션의 필요 증거금은 한도 이하 (무포지션 제외)."""
    for target in [(1, h) for h in range(6)] + [(0, 0)]:
        eng = RiskEngine(flatten_on_stop=flatten)
        reg, h = eng.adjust(*target, equity=equity, start_equity=50_000_000, equity0=50_000_000,
                            reg_price=500.0, margin=MARGIN)
        need = MARGIN.required(reg, h, 500.0)
        assert need == 0 or need <= 0.40 * equity


def test_unchanged_when_no_limit_is_hit():
    """한도에 닿지 않으면 목표를 그대로 둔다 (기존 실험 결과가 바뀌지 않는 조건)."""
    eng = RiskEngine()
    for target in [(1, h) for h in range(6)] + [(0, 0)]:
        assert eng.adjust(*target, equity=50_000_000, start_equity=50_000_000, equity0=50_000_000,
                          reg_price=500.0, margin=MARGIN) == target
    assert eng.log == []


# ---------------------------------------------------------------- P2-01
def ok(price, vol=100):
    return {"futs_prpr": str(price), "acml_vol": str(vol)}


def test_quote_checks_follow_active_legs():
    a_exp = date(2026, 11, 12)
    prev = {"reg": 900.0, "a": 899.0, "b": 901.0}
    bad_b = {"reg": ok(900), "a": ok(899), "b": ok(901, vol=0)}
    bad_a = {"reg": ok(900), "a": {}, "b": ok(901)}
    # 만기 전: a 는 매도 다리라 검사, 아직 쓰지 않는 b 의 거래량 0 은 차단 사유가 아님
    assert quote_problem(date(2026, 10, 20), a_exp, bad_b, prev) is None
    assert quote_problem(date(2026, 10, 20), a_exp, bad_a, prev).startswith("미니 a")
    # a 만기일: a 결제와 b 롤오버가 함께 일어나므로 둘 다 검사
    assert quote_problem(a_exp, a_exp, bad_b, prev).startswith("미니 b")
    assert quote_problem(a_exp, a_exp, bad_a, prev).startswith("미니 a")
    # a 만기 후: b 만 검사
    assert quote_problem(date(2026, 11, 20), a_exp, bad_a, prev) is None
    assert quote_problem(date(2026, 11, 20), a_exp, bad_b, prev).startswith("미니 b")
    # 정규는 항상
    assert quote_problem(date(2026, 10, 20), a_exp, {**bad_b, "reg": {}}, prev).startswith("정규")


# ---------------------------------------------------------------- P1-01
class FixedModel:
    """정해진 행동만 내는 가짜 모델 (predict 인자 형태만 맞춘다)."""

    def __init__(self, action):
        self.action = action


def predict(model, env, obs):
    return model.action


def toy_history():
    """정규 2026-12 + 미니 a 2026-11 + 미니 b 2026-12 의 가짜 일봉 (가격은 교육용 가상 수치)."""
    days = pd.bdate_range("2026-09-28", "2026-10-30")
    n = len(days)
    base = 900 + np.cumsum(np.full(n, 0.5))

    def frame(px):
        return pd.DataFrame({"close": px, "high": px + 2, "low": px - 2, "volume": 1000}, index=days)

    return {"reg": frame(base), "a": frame(base - 0.4), "b": frame(base + 0.2)}, days


CFG = {"mode": "R1", "reward": {"scale": 100.0}}


def run_day(book, model, full, day):
    hist = {k: v[v.index < day] for k, v in full.items()}
    row = full["reg"].loc[day]
    quotes = {"reg_bar": (row.close, row.high, row.low), "a": full["a"].close[day], "b": full["b"].close[day],
              "codes": {"reg": "R", "a": "A", "b": "B"}}
    return run_once(model, CFG, hist, day.date(), quotes, date(2026, 11, 12), date(2026, 12, 10), book, predict,
                    kill_file="__no_kill__", source="replay")


def test_missed_operating_day_holds_previous_position(tmp_path):
    """검수 재현: 10-13까지 (1, 5)로 운영, 10-14 누락, 10-15에 목표 0을 내는 모델로 재실행."""
    full, days = toy_history()
    book = PaperBook(str(tmp_path), "2026-12")
    for day in days[(days >= "2026-10-12") & (days <= "2026-10-13")]:
        run_day(book, FixedModel(5), full, day)            # 운영 시작 전 날짜는 이때 백필됨
    rec = run_day(book, FixedModel(0), full, pd.Timestamp("2026-10-15"))

    logged = book.decisions()
    assert logged["2026-10-14"]["source"] == "missed"
    assert logged["2026-10-14"]["final"] == [1, 5]          # 하지 않은 헤지 해제가 생기지 않음
    assert rec["current"] == [1, 5]                         # 오늘 결정 직전 포지션도 (1, 5)
    assert logged["2026-10-09"]["source"] == "backfill"     # 운영 시작(10-12) 전은 여전히 모델 백필


# ---------------------------------------------------------------- P2-02
def test_ledger_reports_missing_daily_bar(tmp_path):
    full, days = toy_history()
    root = tmp_path / "data"
    (root / "daily").mkdir(parents=True)
    for k, code in {"reg": "A01612", "a": "A05611", "b": "A05612"}.items():
        df = full[k][full[k].index <= "2026-10-14"].reset_index(names="date")
        df["date"] = df["date"].dt.strftime("%Y%m%d")
        df.to_parquet(root / "daily" / f"{code}.parquet", index=False)
    decisions = {"2026-10-15": {"final": [1, 5]}}            # 일봉에 없는 날짜
    with pytest.raises(LedgerDataMissing, match="collect_real"):
        ledger_from_decisions(str(root), 2026, 12, decisions)


def test_paperbook_roundtrip(tmp_path):
    book = PaperBook(str(tmp_path), "x")
    book.append(book.decisions_path, {"date": "2026-01-02", "final": [1, 5]})
    assert json.loads(open(book.decisions_path, encoding="utf-8").read())["final"] == [1, 5]
