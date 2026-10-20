"""CalendarSpreadEnv — 정규 1계약 매수 + 근월 미니 매도의 헤지비율을 고르는 Gymnasium 환경 (5장~).

- 한 에피소드 = 표준 쌍 하나 (hedgelab.episode)
- 행동: 목표 미니 매도 계약 수 0~5 (R1). mode="R2" 이면 행동 6 = 전체 청산(정규 0, 미니 0)
- 관측: 오늘까지의 시장·계좌 정보로 만든 특징 벡터 (미래 정보 없음)
- 보상: 로그 순자산 변화 − 위험 페널티 (페널티 계수는 기본 0, 7장에서 켠다)
회계는 4장의 EpisodeSim 을 그대로 쓴다.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import MINI_MULT, REG_MULT
from .backtest import CostModel, EpisodeSim, MarginModel

FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",      # 정규 로그수익률 (%)
    "vol_20d", "atr_14",                # 실현 변동성, ATR / 종가 (%)
    "ma20_gap",                         # 종가 / 20일 이동평균 − 1 (%)
    "spread_bp", "spread_chg_5d",       # (정규 − 근월 미니) / 종가 (bp), 5일 변화
    "days_to_mini", "days_to_reg",      # 근월 미니·정규 만기까지 남은 거래일 / 20
    "after_a",                          # 미니 a 만기 이후 여부
    "h_now", "reg_now",                 # 현재 미니 매도 계약 수 / 5, 정규 보유 여부
    "ep_pnl", "drawdown", "margin_use",  # 에피소드 손익·낙폭 (시작 자본 대비 %), 증거금 사용률
]
# 관측값을 대략 −10~10 안에 두기 위한 나눗셈 계수 (실제·합성 데이터의 1~99% 분위로 정함)
SCALE = {"ret_5d": 2.0, "ret_20d": 2.0, "spread_bp": 10.0, "spread_chg_5d": 10.0}


@dataclass
class RewardConfig:
    scale: float = 100.0           # 로그수익률 × 100 ≈ % 단위
    lam_dd: float = 0.0            # λ1 낙폭 증가 페널티 (% 단위 낙폭 증가분에 곱함)
    lam_margin: float = 0.0        # λ2 증거금 사용률이 기준을 넘은 만큼에 곱함
    margin_threshold: float = 0.3
    lam_turnover: float = 0.0      # λ3 포지션 변경량(미니 계약/5 + 정규 계약)에 곱함


def market_features(ep, t):
    """오늘(t)까지의 행만으로 시장 특징을 계산한다."""
    hist = ep.iloc[: t + 1]
    close = hist.reg.to_numpy()
    lr = np.diff(np.log(close)) if len(close) > 1 else np.zeros(1)

    def ret(n):
        return 100 * np.log(close[-1] / close[max(0, len(close) - 1 - n)])

    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(hist.reg_high - hist.reg_low,
                    np.maximum(abs(hist.reg_high.to_numpy() - prev), abs(hist.reg_low.to_numpy() - prev)))
    row = hist.iloc[-1]
    near = "b" if row["a_last"] else row["near"]
    spread = (hist.reg - np.where(hist.near == "a", hist.mini_a, hist.mini_b)).to_numpy()
    spread_now = row["reg"] - (row["mini_b"] if near == "b" else row["mini_a"])
    a_idx = np.flatnonzero(ep.a_last.to_numpy())
    days_to_a = (a_idx[0] - t) if (len(a_idx) and t <= a_idx[0]) else 0
    return {
        "ret_1d": 100 * lr[-1] if len(close) > 1 else 0.0,
        "ret_5d": ret(5), "ret_20d": ret(20),
        "vol_20d": 100 * np.std(lr[-20:]) if len(lr) > 1 else 0.0,
        "atr_14": 100 * np.mean(tr[-14:]) / close[-1],
        "ma20_gap": 100 * (close[-1] / np.mean(close[-20:]) - 1),
        "spread_bp": 1e4 * spread_now / close[-1],
        "spread_chg_5d": 1e4 * (spread[-1] - spread[max(0, len(spread) - 6)]) / close[-1],
        "days_to_mini": (days_to_a if near == "a" and not row["a_last"] else len(ep) - 1 - t) / 20,
        "days_to_reg": (len(ep) - 1 - t) / 20,
        "after_a": float(near == "b"),
    }


class CalendarSpreadEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episodes, mode="R1", reward=None, equity0=50_000_000,
                 costs=None, margin=None, random_episode=True, risk=None):
        super().__init__()
        if not episodes:
            raise ValueError("에피소드가 비어 있습니다")
        self.episodes, self.mode = episodes, mode
        self.risk = risk                    # hedgelab.risk.RiskEngine (7장). None 이면 검사 없음
        self.reward_cfg = reward or RewardConfig()
        self.equity0, self.costs, self.margin = equity0, costs or CostModel(), margin or MarginModel()
        self.random_episode = random_episode
        self.action_space = spaces.Discrete(7 if mode == "R2" else 6)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(len(FEATURES),), dtype=np.float32)
        self._next = 0
        # 시장 특징은 행동과 무관하므로 에피소드마다 한 번만 계산해 둔다 (학습 속도 수십 배 차이)
        self._market = [[market_features(ep, t) for t in range(len(ep))] for ep in episodes]

    # ---------------------------------------------------------------- 행동 → 목표 포지션
    def to_target(self, action):
        action = int(action)
        if self.mode == "R2" and action == 6:
            return 0, 0                     # 전체 청산
        return 1, action                    # 정규 1계약 + 미니 action계약 매도

    # ---------------------------------------------------------------- 관측
    def _obs(self):
        f = dict(self._market[self.ep_index][self.sim.t])
        eq = self.sim.marked_equity
        f.update({
            "h_now": self.sim.h / 5, "reg_now": float(self.sim.n_reg),
            "ep_pnl": 100 * (eq - self.start_equity) / self.equity0,
            "drawdown": 100 * (eq - self.peak) / self.equity0,
            "margin_use": self.sim.margin.required(self.sim.n_reg, self.sim.h, self.sim.rows[self.sim.t]["reg"]) / max(eq, 1),
        })
        v = np.array([f[k] / SCALE.get(k, 1.0) for k in FEATURES], dtype=np.float32)
        return np.clip(np.nan_to_num(v), -10.0, 10.0)

    # ---------------------------------------------------------------- Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "episode_index" in options:
            i = options["episode_index"]
        elif self.random_episode:
            i = int(self.np_random.integers(len(self.episodes)))
        else:
            i, self._next = self._next, (self._next + 1) % len(self.episodes)
        self.ep_index, self.ep = i, self.episodes[i]
        self.sim = EpisodeSim(self.ep, self.equity0, self.costs, self.margin)
        self.start_equity = self.peak = self.equity0
        if self.risk:
            self.risk.reset()
        return self._obs(), {"episode": f"{self.ep.attrs['year']}-{self.ep.attrs['month']:02d}"}

    def _risk_args(self):
        return dict(equity=self.sim.marked_equity, start_equity=self.start_equity, equity0=self.equity0,
                    reg_price=self.sim.rows[self.sim.t]["reg"], margin=self.sim.margin)

    def action_masks(self):
        """MaskablePPO 용: 위험관리 규칙을 어기지 않는 행동만 True (7장)."""
        if not self.risk or not hasattr(self, "sim"):
            return np.ones(self.action_space.n, dtype=bool)
        args = self._risk_args()
        mask = np.array([self.risk.allowed(*self.to_target(a), **args) for a in range(self.action_space.n)])
        return mask if mask.any() else np.ones(self.action_space.n, dtype=bool)

    def step(self, action, forced=None):
        """forced=(정규, 미니) 를 주면 그 포지션을 그대로 실행한다 (9장: 이미 확정된 운영 기록 재생용).

        이때 위험관리 엔진은 포지션을 고치지 않고 손실 한도 상태만 갱신한다.
        """
        cfg = self.reward_cfg
        before = self.sim.marked_equity
        old_reg, old_h = self.sim.n_reg, self.sim.h
        adjusted = False
        if forced is not None:
            target_reg, target_h = forced
            if self.risk:
                self.risk.observe(self.sim.marked_equity, self.start_equity, self.equity0)
        else:
            target_reg, target_h = self.to_target(action)
            if self.risk:                   # 위험관리 엔진이 정책보다 우선한다
                fixed = self.risk.adjust(target_reg, target_h, date=self.ep.index[self.sim.t], **self._risk_args())
                adjusted = fixed != (target_reg, target_h)
                target_reg, target_h = fixed
        self.sim.act(target_reg, target_h)
        after = self.sim.equity if self.sim.done else self.sim.marked_equity

        reward = cfg.scale * np.log(max(after, 1.0) / max(before, 1.0))
        dd_before = max(0.0, self.peak - before)
        self.peak = max(self.peak, after)
        dd_after = max(0.0, self.peak - after)
        dd_increase = 100 * max(0.0, dd_after - dd_before) / self.equity0
        rec = self.sim.records[-2] if self.sim.done else self.sim.records[-1]
        margin_use = rec["margin_req"] / max(after, 1.0)
        turnover = abs(target_h - old_h) / 5 + abs(target_reg - old_reg)
        penalty = (cfg.lam_dd * dd_increase
                   + cfg.lam_margin * max(0.0, margin_use - cfg.margin_threshold)
                   + cfg.lam_turnover * turnover)
        reward -= penalty

        terminated = self.sim.done
        obs = np.zeros(len(FEATURES), dtype=np.float32) if terminated else self._obs()
        info = {"equity": after, "h": target_h, "n_reg": target_reg, "penalty": penalty, "risk_adjusted": adjusted,
                "net_delta": REG_MULT * target_reg - MINI_MULT * target_h}
        return obs, float(reward), terminated, False, info
