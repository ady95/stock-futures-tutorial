"""기준선 전략 (4장). 모두 policy(ep, t, state) -> (정규 계약 수, 근월 미니 매도 계약 수) 형태."""
import numpy as np


def b0_cash(ep, t, state):
    """B0 무거래·현금 보유."""
    return 0, 0


def b1_fixed(ep, t, state):
    """B1 고정 1:5 헤지. 미니가 만기되면 다음 근월 미니로 다시 5계약 매도(롤오버)."""
    return 1, 5


def b1n_fixed_no_roll(ep, t, state):
    """B1n 고정 1:5 헤지, 롤오버 없음. 미니 a 만기 후에는 정규 매수만 남는다 (02-4의 공백 구간)."""
    row = ep.iloc[t]
    return (1, 0) if (row["near"] == "b" or row["a_last"]) else (1, 5)


def make_b2_atr(ma=20, atr_n=14, atr_lookback=60, calm=1.2, partial_h=2):
    """B2 ATR·이동평균 규칙형 헤지.

    - 종가가 ma일 이동평균 위(상승 추세)이고
    - ATR이 최근 atr_lookback일 ATR 중앙값의 calm배 미만(변동성 안정)이면
      → 헤지를 일부 풀어 partial_h 계약만 매도 (매수 노출을 남김)
    - 그 밖에는 완전 헤지(5계약)
    지표는 에피소드 안의 과거 데이터만으로 계산한다. 초반 데이터가 모자라면 완전 헤지.
    """
    def policy(ep, t, state):
        hist = ep.iloc[: t + 1]
        if len(hist) < max(ma, atr_n) + 1:
            return 1, 5
        close = hist.reg
        prev = close.shift(1)
        tr = np.maximum(hist.reg_high - hist.reg_low,
                        np.maximum(abs(hist.reg_high - prev), abs(hist.reg_low - prev)))
        atr = tr.rolling(atr_n).mean()
        trend_up = close.iloc[-1] > close.rolling(ma).mean().iloc[-1]
        calm_now = atr.iloc[-1] < calm * atr.iloc[-atr_lookback:].median()
        return (1, partial_h) if (trend_up and calm_now) else (1, 5)
    return policy


BASELINES = {
    "B0 무거래": b0_cash,
    "B1 고정 헤지(롤오버)": b1_fixed,
    "B1n 고정 헤지(롤오버 없음)": b1n_fixed_no_roll,
    "B2 ATR 규칙형": make_b2_atr(),
}
