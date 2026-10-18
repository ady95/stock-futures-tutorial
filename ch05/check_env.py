"""05-5 실습: CalendarSpreadEnv 구현 확인.

실행:  python ch05/check_env.py
1) Gymnasium·Stable-Baselines3 환경 검사기 통과 여부
2) 관측 벡터 한 개 살펴보기
3) 고정 행동(항상 5)으로 돌린 환경의 손익이 4장 백테스트 엔진(B1)과 같은지
4) 무작위 정책으로 몇 에피소드 돌려 보기
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium.utils.env_checker import check_env as gym_check  # noqa: E402
from stable_baselines3.common.env_checker import check_env as sb3_check  # noqa: E402

from hedgelab.backtest import run_episode  # noqa: E402
from hedgelab.env import FEATURES, CalendarSpreadEnv  # noqa: E402
from hedgelab.episode import build_all  # noqa: E402
from hedgelab.strategies import b1_fixed  # noqa: E402

episodes = build_all("data/synthetic", 2020, 2025, synthetic=True)
env = CalendarSpreadEnv(episodes)

# 1) 환경 검사기
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gym_check(env, skip_render_check=True)
    sb3_check(env, warn=False)
print("환경 검사 통과: gymnasium + stable-baselines3")
print(f"행동 공간: {env.action_space}, 관측 공간: {env.observation_space.shape}")

# 2) 관측 벡터
obs, info = env.reset(options={"episode_index": 10})
obs, *_ = env.step(5)
for _ in range(19):
    obs, *_ = env.step(5)
print(f"\n에피소드 {info['episode']}, 20번째 날 관측값")
for name, value in zip(FEATURES, obs):
    print(f"  {name:<14} {value:8.3f}")

# 3) 환경 회계 = 백테스트 엔진 회계?
diffs = []
for i, ep in enumerate(episodes):
    obs, _ = env.reset(options={"episode_index": i})
    done = False
    while not done:
        obs, reward, done, _, info = env.step(5)
    bt = run_episode(ep, b1_fixed)
    diffs.append(info["equity"] - bt.equity.iloc[-1])
print(f"\n고정 행동 5 vs 백테스트 B1: 에피소드 {len(diffs)}개, 최종 순자산 차이 최대 {max(map(abs, diffs)):.6f}원")

# 4) 무작위 정책
rng = np.random.default_rng(0)
returns, lengths = [], []
for _ in range(20):
    obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
    done, total, n = False, 0.0, 0
    while not done:
        obs, reward, done, _, info = env.step(env.action_space.sample())
        total += reward
        n += 1
    returns.append(total)
    lengths.append(n)
print(f"무작위 정책 20회: 에피소드 길이 평균 {np.mean(lengths):.1f}일, "
      f"보상 합 평균 {np.mean(returns):.3f}, 최소 {np.min(returns):.3f}, 최대 {np.max(returns):.3f}")
