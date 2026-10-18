"""06-2 실습: PPO 학습.

실행:  python ch06/train_ppo.py configs/ppo_r1_synthetic.yaml [시드...]
결과:  models/<이름>_seed<N>.zip (+ 설정 yaml)
       runs/<이름>_seed<N>_1/      TensorBoard 로그
       results/<이름>_seed<N>_curve.csv   학습 중 주기적 평가 (학습·평가 구간 보상 합 평균)
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from hedgelab.rl import load_config, load_episodes, make_env, model_class, predict, save_run  # noqa: E402


def episode_returns(model, env, n):
    """평가 환경의 에피소드를 순서대로 n개 돌려 결정적 정책의 보상 합 평균을 낸다."""
    totals = []
    for i in range(n):
        obs, _ = env.reset(options={"episode_index": i})
        done, total = False, 0.0
        while not done:
            obs, r, done, _, _ = env.step(predict(model, env, obs))
            total += r
        totals.append(total)
    return float(np.mean(totals))


class CurveCallback(BaseCallback):
    """every 스텝마다 학습·평가 구간 성과를 기록한다."""

    def __init__(self, train_env, test_env, every, path):
        super().__init__()
        self.train_env, self.test_env, self.every, self.path = train_env, test_env, every, path
        self.rows = []

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self.rows.append({
                "steps": self.num_timesteps,
                "train": episode_returns(self.model, self.train_env, len(self.train_env.episodes)),
                "test": episode_returns(self.model, self.test_env, len(self.test_env.episodes)),
            })
        return True

    def _on_training_end(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["steps", "train", "test"])
            w.writeheader()
            w.writerows(self.rows)


cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/ppo_r1_synthetic.yaml")
seeds = [int(s) for s in sys.argv[2:]] or cfg["train"]["seeds"]
train_eps, test_eps = load_episodes(cfg, "train"), load_episodes(cfg, "test")
print(f"[{cfg['name']}] 학습 에피소드 {len(train_eps)}개({sum(len(e) for e in train_eps)}일), "
      f"평가 에피소드 {len(test_eps)}개")

p = dict(cfg["ppo"])
net_arch = p.pop("net_arch")
every = cfg["train"].get("eval_every", 10_000)
os.makedirs("results", exist_ok=True)
for seed in seeds:
    env = make_env(train_eps, cfg)
    if cfg.get("algo") != "maskable":
        env = Monitor(env)                 # MaskablePPO 는 action_masks 를 쓰므로 원래 환경을 그대로 넘긴다
    model = model_class(cfg)("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                policy_kwargs={"net_arch": net_arch}, tensorboard_log="runs", **p)
    curve = CurveCallback(make_env(train_eps, cfg, random_episode=False),
                          make_env(test_eps, cfg, random_episode=False),
                          every, f"results/{cfg['name']}_seed{seed}_curve.csv")
    t0 = time.time()
    model.learn(total_timesteps=cfg["train"]["total_timesteps"], callback=curve,
                tb_log_name=f"{cfg['name']}_seed{seed}")
    path = save_run(model, cfg, f"{cfg['name']}_seed{seed}")
    last = curve.rows[-1]
    print(f"seed {seed}: {time.time() - t0:.0f}초 | 마지막 평가 보상 합 평균 — "
          f"학습 {last['train']:.3f}, 평가 {last['test']:.3f} → {path}.zip")
