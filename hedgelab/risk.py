"""독립 위험관리 엔진 (7장).

정책(PPO 또는 규칙)이 낸 목표 포지션을 실행 전에 검사하고, 규칙을 어기면 고친다.
학습 모델의 판단보다 이 규칙이 항상 우선한다.

규칙 (숫자는 교육용 가정)
  1. 증거금 한도: 필요 증거금 / 순자산 > max_margin_use 이면 헤지를 늘려(미니 매도 추가) 한도 안으로.
     그래도 넘으면 전체 청산
  2. 에피소드 손실 한도: 에피소드 시작 대비 손실이 stop_loss(시작 자본 대비 비율)를 넘으면
     남은 기간은 완전 헤지(R1) 또는 전체 청산(R2)만 허용
"""
from dataclasses import dataclass, field


@dataclass
class RiskEngine:
    max_margin_use: float = 0.40
    stop_loss: float = 0.06
    flatten_on_stop: bool = False        # True: 손실 한도 도달 시 전체 청산 (R2), False: 완전 헤지 (R1)
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.stopped = False

    def reset(self):
        """에피소드마다 호출. 손실 한도 상태를 풀고, 개입 기록은 유지한다."""
        self.stopped = False

    def allowed(self, target_reg, target_h, equity, start_equity, equity0, reg_price, margin):
        """목표 포지션이 규칙을 지키는지만 판정 (행동 마스킹용)."""
        adj = self.adjust(target_reg, target_h, equity, start_equity, equity0, reg_price, margin, record=False)
        return adj == (target_reg, target_h)

    def adjust(self, target_reg, target_h, equity, start_equity, equity0, reg_price, margin,
               date=None, record=True):
        """규칙에 맞게 고친 (정규, 미니)를 돌려준다. 고쳤으면 log 에 남긴다."""
        reason = None
        reg, h = target_reg, target_h

        # 규칙 2: 손실 한도 (한 번 걸리면 에피소드 끝까지 유지)
        if self.stopped or (start_equity - equity) > self.stop_loss * equity0:
            if record:
                self.stopped = True
            safe = (0, 0) if self.flatten_on_stop else (1 if reg else 0, 5 if reg else 0)
            if (reg, h) != safe:
                reg, h, reason = safe[0], safe[1], "손실 한도"

        # 규칙 1: 증거금 한도
        if reason is None and margin.required(reg, h, reg_price) > self.max_margin_use * equity:
            while h < 5 and margin.required(reg, h, reg_price) > self.max_margin_use * equity:
                h += 1
            if margin.required(reg, h, reg_price) > self.max_margin_use * equity:
                reg, h = 0, 0
            reason = "증거금 한도"

        if record and reason:
            self.log.append({"date": date, "target": (target_reg, target_h), "adjusted": (reg, h), "reason": reason})
        return reg, h


def with_risk(policy, engine, equity0=50_000_000):
    """4장 백테스트용: 정책의 목표 포지션을 위험관리 엔진에 통과시킨 뒤 실행한다."""
    def wrapped(ep, t, state):
        if t == 0:
            engine.reset()
            wrapped.start = state["equity"]
        target = policy(ep, t, state)
        return engine.adjust(*target, equity=state["equity"], start_equity=wrapped.start, equity0=equity0,
                             reg_price=ep.reg.iloc[t], margin=state["margin"], date=ep.index[t])
    wrapped.start = equity0
    return wrapped
