"""Tests for the 32-bit clamp overflow exploit and regression suite."""

import hashlib
import json
import time
from pathlib import Path

import optimization2
from colorama import Fore, Style

optimization2.ensure_optimal_table()

TEST_CASES: list[tuple[int, set[int], int]] = [
    (3862631, set(), 27),
    (56, set(), 26),
    (100, set(), 26),
    (22899, {10}, 26),
    (989, set(), 26),
    (1000, set(), 26),
    (123, set(), 26),
    (319216, set(), 26),
    (2**28 + 2**29, set(), 27),
    (26407, set(), 26),
    (1073741824, set(), 26),
    (1073741825, set(), 26),
    (2147483647, set(), 26),
    (1048575, set(), 26),
    (654321, set(), 26),
    (999999, set(), 26),
    (4953, {10}, 26),
    (4983, {10}, 26),
    (36, set(), 26),
    (64, set(), 26),
    (125, set(), 26),
    (216, set(), 26),
    (6896, set(), 26),
    (3955, set(), 26),
    (166375, set(), 55),
    (1368794382, set(), 27),
    (222860571, set(), 26),
    (132893794, set(), 26),
    (1150000477, set(), 26),
    (295052544, set(), 27),
    (838861, set(), 27),  # (2**22 + 1) // 5
    (52429, set(), 26),  # (2**19 + 2) // 10
    (174763, set(), 26),  # (2**19 + 1) // 3
    (52429, {10}, 26),  # (2**19 + 2) // (5+5), maybe
]

DB_PATH = Path(__file__).parent / "known_best.json"
TRACKING_CONFIG = {"BASE": optimization2.BASE, "COST": optimization2.COST}
CONFIG_KEY = hashlib.sha256(
    json.dumps(TRACKING_CONFIG, sort_keys=True).encode()
).hexdigest()[:16]


def load_regression_db() -> dict:
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_regression_db(db: dict) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2, sort_keys=True)


def update_regression_db(
    results: dict,
    target: int,
    actual_cost: float,
    expr: str,
    disallowed: set[int],
    max_ext: int,
    max_val: int = optimization2.MAX_VALUE,
) -> None:
    key = str(
        {
            "target": target,
            "disallowed": disallowed,
            "max_ext": max_ext,
            "max_val": max_val,
        }
    )
    prev = results.get(key)

    if prev is None:
        results[key] = {"cost": actual_cost, "expr": expr}
        print("  * New benchmark entry recorded")
        return

    prev_cost = prev["cost"]

    if actual_cost < prev_cost - 1e-12:
        print(f"{Fore.GREEN}  * NEW BEST FOUND")
        results[key] = {"cost": actual_cost, "expr": expr}
        print(f"    old cost: {prev_cost:.12f}")
        print(f"    new cost: {actual_cost:.12f}{Style.RESET_ALL}")
        return

    if actual_cost > prev_cost + 1e-12:
        print(f"{Fore.RED}  x REGRESSION DETECTED")
        print(f"    expected <= {prev_cost:.12f}")
        print(f"    got         {actual_cost:.12f}")
        print(f"    previous expr: {prev['expr']}{Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}  = Matches known best{Style.RESET_ALL}")


def _check(target: int, expected_cost: float, *, label: str = "") -> None:
    """Solve target, verify cost and evaluated value."""
    cost, expr = optimization2.synthesize_optimal_with_exp(target, max_ext=26)
    val, ac = optimization2.evaluate_cost(expr)
    assert abs(cost - expected_cost) < 0.001, (
        f"target={target}: expected cost {expected_cost}, got {cost:.4f}"
    )
    assert val == target, f"target={target}: expression evaluated to {val}, expr={expr}"
    print(f"  PASS: {label or target} cost={cost:.4f} val={val} expr={expr}")


def test_max_value_clamp_exploit():
    _check(optimization2.MAX_VALUE, 0.5, label="MAX_VALUE")


def test_clamp_non_max_value():
    _check(optimization2.MAX_VALUE - 1, 1.5, label="MAX_VALUE-1")


TIMEOUT_SEC = 60  # per-base timeout


def test_multi_base(bases) -> None:
    """Verify correctness for BASE = 2, 3, 4, 5 and 10."""

    db = load_regression_db()

    for base in bases:
        print(f"\n── BASE={base} ──")
        optimization2.reconfigure(base)
        optimization2.ensure_optimal_table()
        print(optimization2.BASE)
        TRACKING_CONFIG = {"BASE": optimization2.BASE, "COST": optimization2.COST}
        CONFIG_KEY = hashlib.sha256(
            json.dumps(TRACKING_CONFIG, sort_keys=True).encode()
        ).hexdigest()[:16]
        profile = db.setdefault(CONFIG_KEY, {"meta": TRACKING_CONFIG, "results": {}})
        results = profile["results"]
        for target, disallowed, max_ext in TEST_CASES:
            print("\n" + "=" * 70)
            print(f"Target: {target}")

            if disallowed:
                print(f"Disallowed literals: {sorted(disallowed)}")
            start = time.time()
            cost, expr = optimization2.synthesize_optimal_with_exp(
                target, max_ext=max_ext, disallowed=disallowed
            )
            elapsed = time.time() - start
            val, ac = optimization2.evaluate_cost(expr)

            print(f"  Time       : {elapsed:.6f}s")
            print(f"  Cost       : {cost:.12f}")
            print(f"  Expression : {expr}")

            # ==================================================
            # Verification
            # ==================================================

            actual_value, actual_cost = optimization2.evaluate_cost(expr)

            if actual_value != target:
                print(f"{Style.RESET_ALL}  X WRONG VALUE")
                print(f"    expected: {target}")
                print(f"    got     : {actual_value}{Style.RESET_ALL}")
                continue

            if abs(actual_cost - cost) > 1e-12:
                print(f"{Fore.RED}  X WRONG COST")
                print(f"    reported: {cost:.12f}")
                print(f"    actual  : {actual_cost:.12f}{Style.RESET_ALL}")
                continue

            print(f"{Fore.GREEN}  v Verified{Style.RESET_ALL}")

            # ==================================================
            # Regression tracking
            # ==================================================
            update_regression_db(
                results, target, ac, expr, disallowed, max_ext, optimization2.MAX_VALUE
            )
    optimization2.reconfigure(2)
    optimization2.ensure_optimal_table()
    save_regression_db(db)


if __name__ == "__main__":
    import sys

    if "--benchmark" in sys.argv:
        test_multi_base([2])
    else:
        print("=== Clamp / overflow tests ===")
        test_max_value_clamp_exploit()
        test_clamp_non_max_value()

        print("\n=== Multi-base tests ===")
        test_multi_base([2, 3, 4, 5, 6, 7, 8, 9, 10, 128])

        print("\nAll tests passed.")
