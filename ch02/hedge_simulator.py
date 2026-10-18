"""02-5 실습: 헤지비율에 따른 손익 시뮬레이터.

정규 코스피200 선물 1계약 매수 + 미니 코스피200 선물 h계약 매도(h = 0~5) 포지션의
손익을 계산한다. 가격 경로는 교육용 가상 수치다.
"""

REG_MULT = 250_000   # 정규선물 거래승수 (원/포인트)
MINI_MULT = 50_000   # 미니선물 거래승수 (원/포인트)


def net_delta(h):
    """지수 1포인트 변화에 대한 순 민감도 (원/포인트)."""
    return REG_MULT * 1 - MINI_MULT * h


def position_pnl(d_reg, d_mini, h):
    """정규 1계약 매수 + 미니 h계약 매도의 손익.

    d_reg: 정규선물 가격 변화 (포인트)
    d_mini: 미니선물 가격 변화 (포인트)
    """
    return REG_MULT * d_reg * 1 - MINI_MULT * d_mini * h


def path_pnl(reg_path, mini_path, h):
    """가격 경로를 따라간 누적 손익 목록. mini_path 의 None 은 미니가 만기로 사라진 시점."""
    out, total = [], 0
    for i in range(1, len(reg_path)):
        d_reg = reg_path[i] - reg_path[i - 1]
        if mini_path[i] is None or mini_path[i - 1] is None:
            d_mini, hh = 0, 0          # 미니 만기 이후: 매도 포지션 없음
        else:
            d_mini, hh = mini_path[i] - mini_path[i - 1], h
        total += position_pnl(d_reg, d_mini, hh)
        out.append(int(round(total)))   # 부동소수점 오차(-0.0 등) 제거
    return out


def main():
    print("실습 1: 미니 매도 계약 수에 따른 순 민감도")
    print(f"{'미니 매도':>8} {'헤지비율':>8} {'순 민감도(원/pt)':>16}")
    for h in range(6):
        print(f"{h:>7}계약 {h / 5:>8.0%} {net_delta(h):>16,}")

    print("\n실습 2: 시나리오별 손익 (원)")
    scenarios = [
        ("두 월물 모두 +10pt", 10.0, 10.0),
        ("두 월물 모두 -10pt", -10.0, -10.0),
        ("정규 +10pt, 미니 +8pt (스프레드 +2)", 10.0, 8.0),
        ("정규 -10pt, 미니 -8pt (스프레드 -2)", -10.0, -8.0),
    ]
    print(f"{'시나리오':<28}" + "".join(f"{f'h={h}':>13}" for h in (0, 3, 5)))
    for name, d_reg, d_mini in scenarios:
        row = "".join(f"{position_pnl(d_reg, d_mini, h):>13,.0f}" for h in (0, 3, 5))
        print(f"{name:<28}{row}")

    print("\n실습 3: 가격 경로 (교육용 가상 수치) — 4일째 미니 만기, 이후 정규만 남음")
    reg = [880.00, 872.00, 861.50, 858.00, 845.00, 830.50, 838.00]
    mini = [879.60, 871.70, 861.10, 857.80, None, None, None]
    print(f"{'일':>3} {'정규':>8} {'미니':>8}" + "".join(f"{f'누적 h={h}':>14}" for h in (0, 5)))
    p0, p5 = path_pnl(reg, mini, 0), path_pnl(reg, mini, 5)
    print(f"{0:>3} {reg[0]:>8.2f} {mini[0]:>8.2f}{0:>14,}{0:>14,}")
    for i in range(1, len(reg)):
        m = f"{mini[i]:>8.2f}" if mini[i] is not None else f"{'만기':>6}"
        print(f"{i:>3} {reg[i]:>8.2f} {m}{p0[i - 1]:>14,.0f}{p5[i - 1]:>14,.0f}")


if __name__ == "__main__":
    main()
