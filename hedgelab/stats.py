"""쌍 단위 통계 — 실제 데이터와 합성 데이터를 같은 잣대로 비교한다 (02-3, 02-4, 03-5)."""
import numpy as np
import pandas as pd

from . import MINI_MULT, REG_MULT
from .data import align_pair, hedge_window


def pair_stats(reg, mini, days=45):
    """헤지 구간(미니 만기 전 days일) 통계와 공백 구간(미니 만기 → 정규 만기) 통계.

    reg, mini: 날짜 인덱스 일봉 DataFrame (close, volume 컬럼)
    """
    mini_exp = mini.index.max()
    w = hedge_window(align_pair(reg, mini), mini_exp, days)
    if len(w) < 10:
        return None
    d = w[["reg", "mini"]].diff().dropna()
    naked = REG_MULT * d.reg
    hedged = REG_MULT * d.reg - 5 * MINI_MULT * d.mini
    gap = reg.loc[reg.index >= mini_exp, "close"]
    gap_pnl = REG_MULT * (gap - gap.iloc[0]) if len(gap) >= 2 else pd.Series([0.0])
    return {
        "days": len(w),
        "level": w.reg.iloc[-1],
        "daily_vol_pct": float((w.reg.pct_change().std()) * 100),
        "corr": float(d.reg.corr(d.mini)),
        "spread_mean": float(w.spread.mean()),
        "spread_std": float(w.spread.std()),
        "risk_ratio": float(hedged.std() / naked.std()),
        "gap_days": len(gap) - 1,
        "gap_pnl": float(gap_pnl.iloc[-1]),
        "gap_worst": float(gap_pnl.min()),
    }


def summarize(rows):
    """pair_stats 결과 목록을 중앙값 위주로 요약한다."""
    df = pd.DataFrame([r for r in rows if r])
    return {
        "쌍 수": len(df),
        "일간 변동성 중앙값(%)": round(df.daily_vol_pct.median(), 2),
        "일간 변화 상관 중앙값": round(df["corr"].median(), 4),
        "스프레드 평균의 범위(pt)": f"{df.spread_mean.min():.2f} ~ {df.spread_mean.max():.2f}",
        "쌍 내 스프레드 표준편차 중앙값(pt)": round(df.spread_std.median(), 2),
        "위험 비율 중앙값(헤지/무헤지)": round(df.risk_ratio.median(), 3),
        "공백 구간 손실 비율": f"{int((df.gap_pnl < 0).sum())}/{len(df)}",
        "공백 구간 최악 시점 중앙값(원)": int(np.median(df.gap_worst)),
    }
