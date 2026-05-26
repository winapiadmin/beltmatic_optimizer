# Hinting for optimization2
from __future__ import annotations
import math

from dataclasses import dataclass
from functools import cache
from optimization2 import (
    BASE,
    COST,
    ensure_optimal_table,
    optimal_lookup,
    parse_expr,
)


@dataclass(slots=True, frozen=True)
class Expr:
    value: int
    cost: float
    text: str


def synthesize_dp(
    target: int,
    *,
    max_value: int = (1 << 31) - 1,
    max_literal: int = 26,
    beam_size: int = 200,
    iterations: int = 50,
    allow_pow: bool = True,
) -> Expr | None:
    """
    Beam-search dynamic programming arithmetic synthesizer.

    Keeps only the most promising expressions
    instead of exploding quadratically forever
    like a doomed astrophysics simulation.
    """

    best: dict[int, Expr] = {}

    def add(expr: Expr) -> bool:
        old = best.get(expr.value)

        if old is None or expr.cost < old.cost:
            best[expr.value] = expr
            return True

        return False

    # --------------------------------------------------
    # Seed literals
    # --------------------------------------------------

    for x in range(max_literal + 1):
        add(
            Expr(
                value=x,
                cost=COST["lit"],
                text=str(x),
            )
        )

    # --------------------------------------------------
    # Seed BASE powers
    # --------------------------------------------------

    p = BASE
    k = 1

    while p <= max_value:
        add(
            Expr(
                value=p,
                cost=COST["powbase"],
                text=f"{BASE}**{k}",
            )
        )

        p *= BASE
        k += 1

    # --------------------------------------------------
    # Main DP loop
    # --------------------------------------------------

    for _ in range(iterations):
        if len(best) == max_value:
            break

        # Most promising expressions only
        frontier = sorted(
            best.values(),
            key=lambda e: (
                e.cost,
                min(
                    abs(target - e.value),
                    abs(target - e.value * e.value),  # useful pow anchor
                    abs(target - e.value * 2),  # useful mul anchor
                ),
            ),
        )[:beam_size]

        changed = False

        for i, a in enumerate(frontier):
            for b in frontier[i:]:

                # ==========================================
                # ADD
                # ==========================================

                nv = a.value + b.value

                if nv <= max_value:

                    expr = Expr(
                        value=nv,
                        cost=a.cost + b.cost + COST["add"],
                        text=f"({a.text}+{b.text})",
                    )

                    changed |= add(expr)

                # ==========================================
                # MUL
                # ==========================================

                nv = a.value * b.value

                if nv <= max_value:

                    expr = Expr(
                        value=nv,
                        cost=a.cost + b.cost + COST["mul"],
                        text=f"({a.text}*{b.text})",
                    )

                    changed |= add(expr)

                # ==========================================
                # SUB
                # ==========================================

                if a.value > b.value:

                    nv = a.value - b.value

                    expr = Expr(
                        value=nv,
                        cost=a.cost + b.cost + COST["sub"],
                        text=f"({a.text}-{b.text})",
                    )

                    changed |= add(expr)

                # ==========================================
                # POW
                # ==========================================

                if (
                    allow_pow
                    and a.value > 1
                    and b.value > 1
                    and b.value <= 8
                    and a.value <= 64
                ):
                    try:
                        nv = a.value**b.value
                    except OverflowError:
                        continue

                    if nv <= max_value:

                        expr = Expr(
                            value=nv,
                            cost=a.cost + b.cost + COST["pow"],
                            text=f"({a.text}**{b.text})",
                        )

                        changed |= add(expr)
        # Converged
        if not changed:
            break

    return best


@cache
def precomp(target, max_val, max_lit):
    result2 = {
        k: (v.cost, parse_expr(v.text))
        for k, v in synthesize_dp(
            target,
            max_value=max_val,
            max_literal=max_lit,
            beam_size=250 * max(max_val // 1000, 1),
            iterations=60 * max(max_val // 100, 1),
        ).items()
    }
    return result2


# ======================================================
# Demo
# ======================================================

if __name__ == "__main__":
    print(precomp(100, 200, 26))
    TESTS = [
        36,
        56,
        64,
        100,
        123,
        125,
        256,
        989,
        1000,
        3955,
        4953,
        4983,
        6896,
        22899,
    ]

    for target in TESTS:

        result = synthesize_dp(
            target,
            max_literal=26,
            max_value=target * 5,
            beam_size=250 * max(target // 1000, 1),
            iterations=60 * max(target // 100, 1),
        ).get(target)

        print("=" * 70)
        print("target:", target)

        if result is None:
            print("not found")
            continue

        print("cost :", round(result.cost, 4))
        print("expr :", result.text)
        print("value:", eval(result.text))
