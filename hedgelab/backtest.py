"""경량 백테스트 엔진 (4장). 5장의 강화학습 환경도 이 회계를 그대로 쓴다.

하루의 처리 순서 (장 마감 시점 t)
  1. 평가손익: 어제 마감 때 들고 있던 포지션을 오늘 종가로 평가한다
  2. 미니 a 최종결제일이면 a 매도 포지션이 결제되어 사라진다 (비용 없음)
  3. 정규 만기일이면 모두 결제하고 에피소드를 끝낸다
  4. 정책이 목표 포지션(정규 계약 수, 근월 미니 매도 계약 수)을 정한다 — 오늘까지의 정보만 쓴다
  5. 목표와의 차이만큼 오늘 종가에 거래한다 (수수료 + 슬리피지)

비용·증거금 숫자는 교육용 가정이다. 실제 수수료와 증거금 제도(거래소 산출 방식)는 다르다 (04-3, 04-4).
"""
from dataclasses import dataclass

import pandas as pd

from . import MINI_MULT, MINI_TICK, REG_MULT, REG_TICK


@dataclass
class CostModel:
    fee_rate: float = 0.00003    # 체결 금액 대비 수수료 (편도 0.003%, 가정)
    slip_ticks: float = 1.0      # 계약당 슬리피지 (호가 칸 수, 가정)

    def reg(self, contracts, price):
        return abs(contracts) * (self.fee_rate * price * REG_MULT + self.slip_ticks * REG_TICK * REG_MULT)

    def mini(self, contracts, price):
        return abs(contracts) * (self.fee_rate * price * MINI_MULT + self.slip_ticks * MINI_TICK * MINI_MULT)


@dataclass
class MarginModel:
    rate: float = 0.10           # 순노출 명목가치 대비 증거금률 (가정)
    spread_rate: float = 0.02    # 서로 상쇄된(헤지된) 명목가치에 대한 증거금률 (가정)
    maintenance: float = 2 / 3   # 유지증거금 = 증거금 × 2/3 (가정)

    def required(self, n_reg, h, reg_price):
        long_notional = n_reg * reg_price * REG_MULT
        short_notional = h * reg_price * MINI_MULT          # 미니 가격 ≈ 정규 가격으로 근사
        hedged = min(long_notional, short_notional)
        net = abs(long_notional - short_notional)
        return self.rate * net + self.spread_rate * hedged


def mini_price(row, which):
    return row["mini_a"] if which == "a" else row["mini_b"]


def run_episode(ep, policy, equity0=50_000_000, costs=None, margin=None):
    """에피소드 하나를 돌려 일별 기록 DataFrame 을 돌려준다.

    policy(ep, t, state) -> (n_reg, h)
      ep: 에피소드 DataFrame, t: 오늘의 위치(정수), state: 현재 포지션·계좌 정보 dict
      ep.iloc[:t + 1] 까지만 봐야 한다 (미래 정보 금지)
    """
    costs, margin = costs or CostModel(), margin or MarginModel()
    n_reg, h, held = 0, 0, None            # held: 매도 중인 미니 ('a'/'b')
    equity, records = equity0, []
    rows = ep.to_dict("records")
    for t, row in enumerate(rows):
        pnl_reg = pnl_mini = pnl_dir = pnl_spread = cost = 0.0
        if t > 0:
            prev = rows[t - 1]
            d_reg = row["reg"] - prev["reg"]
            d_mini = (mini_price(row, held) - mini_price(prev, held)) if h else 0.0
            pnl_reg = REG_MULT * n_reg * d_reg
            pnl_mini = -MINI_MULT * h * d_mini
            # 손익 분해: 방향 = 순노출 × 정규 변화, 스프레드 = 헤지된 부분 × (정규 변화 − 미니 변화)
            pnl_dir = (REG_MULT * n_reg - MINI_MULT * h) * d_reg
            pnl_spread = MINI_MULT * h * (d_reg - d_mini)
        if row["a_last"] and held == "a":
            h, held = 0, None              # a 최종결제: 오늘 종가로 정산 (비용 없음)

        last = t == len(rows) - 1
        if last:
            target_reg, target_h = 0, 0    # 정규·b 만기 결제
        else:
            state = {"n_reg": n_reg, "h": h, "equity": equity + pnl_reg + pnl_mini, "equity0": equity0}
            target_reg, target_h = policy(ep, t, state)
        near = "b" if row["a_last"] else row["near"]
        if not last:
            cost += costs.reg(target_reg - n_reg, row["reg"])
            cost += costs.mini(target_h - h, mini_price(row, near))
        n_reg, h = target_reg, target_h
        held = near if h else None

        equity += pnl_reg + pnl_mini - cost
        records.append({
            "date": ep.index[t], "n_reg": n_reg, "h": h, "near": near,
            "pnl_reg": pnl_reg, "pnl_mini": pnl_mini, "pnl_dir": pnl_dir, "pnl_spread": pnl_spread,
            "cost": cost, "equity": equity,
            "net_delta": REG_MULT * n_reg - MINI_MULT * h,
            "margin_req": margin.required(n_reg, h, row["reg"]),
        })
    out = pd.DataFrame(records).set_index("date")
    out.attrs.update(ep.attrs)
    return out


def run_many(episodes, policy, equity0=50_000_000, **kw):
    """여러 에피소드를 이어서 돌린다. 계좌는 에피소드마다 처음 금액으로 시작하지 않고 이어진다."""
    frames, equity = [], equity0
    for ep in episodes:
        rec = run_episode(ep, policy, equity0=equity, **kw)
        rec["episode"] = f"{ep.attrs['year']}-{ep.attrs['month']:02d}"
        equity = rec.equity.iloc[-1]
        frames.append(rec)
    out = pd.concat(frames)
    out.attrs["equity0"] = equity0
    return out
