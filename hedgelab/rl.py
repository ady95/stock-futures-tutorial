"""PPO 학습·평가 도우미 (6장~)."""
import os

import numpy as np
import pandas as pd
import yaml

from .env import CalendarSpreadEnv, RewardConfig
from .episode import build_all
from .risk import RiskEngine


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_episodes(cfg, split):
    """cfg['data'] 와 split('train'/'test')의 연도 범위로 에피소드 목록을 만든다."""
    d = cfg["data"]
    start, end = d[split]
    return build_all(d["root"], start, end, synthetic=d.get("synthetic", False))


def make_env(episodes, cfg, random_episode=True):
    risk = RiskEngine(**cfg["risk"]) if cfg.get("risk") else None
    return CalendarSpreadEnv(episodes, mode=cfg.get("mode", "R1"),
                             reward=RewardConfig(**cfg.get("reward", {})),
                             random_episode=random_episode, risk=risk)


def model_class(cfg):
    """cfg['algo'] 가 maskable 이면 sb3-contrib 의 MaskablePPO, 아니면 PPO."""
    if cfg.get("algo") == "maskable":
        from sb3_contrib import MaskablePPO
        return MaskablePPO
    from stable_baselines3 import PPO
    return PPO


def predict(model, env, obs):
    """결정적 행동. MaskablePPO 면 환경의 행동 마스크를 함께 넘긴다."""
    if type(model).__name__ == "MaskablePPO":
        action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
    else:
        action, _ = model.predict(obs, deterministic=True)
    return action


def evaluate(model, episodes, cfg, equity0=50_000_000):
    """학습된 모델을 에피소드마다 결정적(deterministic)으로 돌려 일별 기록을 이어 붙인다.

    반환 DataFrame 은 4장 run_many 결과와 같은 열을 가지므로 metrics.summary 에 그대로 넣을 수 있다.
    """
    env = make_env(episodes, cfg, random_episode=False)
    frames = []
    for i in range(len(episodes)):
        obs, info = env.reset(options={"episode_index": i})
        done = False
        while not done:
            obs, _, done, _, _ = env.step(predict(model, env, obs))
        rec = env.sim.frame()
        rec["episode"] = info["episode"]
        frames.append(rec)
    out = pd.concat(frames)
    out["equity"] = equity0 + (out.pnl_reg + out.pnl_mini - out.cost).cumsum()
    out.attrs["equity0"] = equity0
    out.attrs["risk_log"] = env.risk.log if env.risk else []
    return out


def action_profile(rec):
    """미니 매도 계약 수(h) 분포와 구간별 평균 h."""
    dist = rec.h.value_counts(normalize=True).sort_index()
    by_phase = rec.groupby(rec.near).h.mean()
    return dist, by_phase


def save_run(model, cfg, name, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    model.save(path)
    with open(path + ".yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def seed_everything(seed):
    np.random.seed(seed)
