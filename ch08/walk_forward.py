"""08-4 실습: 워크포워드 검증 (실제 데이터).

평가 연도 T 마다
  학습: first_year ~ T−2년 에피소드
  검증: T−1년 에피소드 — 학습 중 eval_every 스텝마다 평가해 가장 좋은 체크포인트를 고른다
  평가: T년 에피소드   — 고른 체크포인트로 한 번만 평가한다
결과는 results/wf/<이름>_<T>_seed<N>.parquet 에 일별 기록으로 저장한다 (08-5 report.py 가 모아서 분석).

실행:  python ch08/walk_forward.py configs/wf_r1.yaml [평가연도...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from hedgelab.episode import build_all  # noqa: E402
from hedgelab.rl import evaluate, load_config, make_env, model_class, predict  # noqa: E402


def mean_return(model, env):
    total = 0.0
    for i in range(len(env.episodes)):
        obs, _ = env.reset(options={"episode_index": i})
        done = False
        while not done:
            obs, r, done, _, _ = env.step(predict(model, env, obs))
            total += r
    return total / len(env.episodes)


class BestOnValidation(BaseCallback):
    """검증 구간 보상 합 평균이 가장 높았던 시점의 모델을 저장한다 (조기 종료와 같은 효과)."""

    def __init__(self, val_env, every, path):
        super().__init__()
        self.val_env, self.every, self.path = val_env, every, path
        self.best, self.best_step = -1e9, 0

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            score = mean_return(self.model, self.val_env)
            if score > self.best:
                self.best, self.best_step = score, self.num_timesteps
                self.model.save(self.path)
        return True


cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/wf_r1.yaml")
wf, root = cfg["walk_forward"], cfg["data"]["root"]
years = [int(y) for y in sys.argv[2:]] or wf["test_years"]
p = dict(cfg["ppo"])
net_arch = p.pop("net_arch")
os.makedirs("results/wf", exist_ok=True)
os.makedirs("models/wf", exist_ok=True)

for T in years:
    train_eps = build_all(root, wf["first_year"], T - 2)
    val_eps, test_eps = build_all(root, T - 1, T - 1), build_all(root, T, T)
    print(f"[{cfg['name']}] 평가 {T}: 학습 {wf['first_year']}~{T - 2} ({len(train_eps)}개), "
          f"검증 {T - 1} ({len(val_eps)}개), 평가 {T} ({len(test_eps)}개)")
    for seed in cfg["train"]["seeds"]:
        out = f"results/wf/{cfg['name']}_{T}_seed{seed}.parquet"
        if os.path.exists(out):
            continue
        model_path = f"models/wf/{cfg['name']}_{T}_seed{seed}"
        model = model_class(cfg)("MlpPolicy", make_env(train_eps, cfg), seed=seed, verbose=0, device="cpu",
                                 policy_kwargs={"net_arch": net_arch}, **p)
        cb = BestOnValidation(make_env(val_eps, cfg, random_episode=False), cfg["train"]["eval_every"], model_path)
        t0 = time.time()
        model.learn(total_timesteps=cfg["train"]["total_timesteps"], callback=cb)
        best = model_class(cfg).load(model_path, device="cpu")
        rec = evaluate(best, test_eps, cfg)
        rec["test_year"], rec["seed"], rec["best_step"] = T, seed, cb.best_step
        rec.to_parquet(out)
        pnl = rec.equity.iloc[-1] - rec.attrs["equity0"]
        print(f"  seed {seed}: {time.time() - t0:.0f}초, 검증 최고 {cb.best:.3f} @ {cb.best_step}스텝, "
              f"평가 {T} 순손익 {pnl:,.0f}원")
