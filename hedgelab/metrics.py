"""성과 지표 (4장, 8장). 계좌 규모가 고정(정규 1계약)이므로 수익률은 시작 자본 대비로 계산한다."""
import numpy as np

from . import REG_MULT


def summary(rec, equity0=None):
    equity0 = equity0 or rec.attrs.get("equity0", 50_000_000)
    pnl = rec.equity.diff().fillna(rec.equity.iloc[0] - equity0)
    curve = equity0 + pnl.cumsum()
    drawdown = curve - curve.cummax().clip(lower=equity0)
    worst = np.sort(pnl.to_numpy())[: max(1, int(len(pnl) * 0.05))]
    trades = rec.n_reg.diff().abs().fillna(rec.n_reg.iloc[0]).sum() + rec.h.diff().abs().fillna(rec.h.iloc[0]).sum()
    std = pnl.std()
    return {
        "순손익(원)": round(curve.iloc[-1] - equity0),
        "수익률(%)": round((curve.iloc[-1] / equity0 - 1) * 100, 2),
        "MDD(원)": round(drawdown.min()),
        "MDD(%)": round(drawdown.min() / equity0 * 100, 2),
        "Sharpe": round(pnl.mean() / std * np.sqrt(252), 2) if std > 0 else 0.0,
        "CVaR5%(원)": round(worst.mean()),
        "최악일(원)": round(pnl.min()),
        "평균 노출(정규 계약)": round((rec.net_delta.abs() / REG_MULT).mean(), 2),
        "거래 계약 수": int(trades),
        "방향 손익(원)": round(rec.pnl_dir.sum()),
        "스프레드 손익(원)": round(rec.pnl_spread.sum()),
        "비용(원)": round(rec.cost.sum()),
        "거래일": len(rec),
    }
