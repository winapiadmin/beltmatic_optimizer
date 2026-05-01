import math
import ast
import bisect
from typing import Optional
from functools import lru_cache

BASE = 2
COST = {
    "add": 1 / 2,
    "sub": 1 / 1,
    "mul": 1 / 1,
    "exp": 1 / 1.1,
    "powbase": 0.00,
    "lit": 0.00,
}


# =========================================================
# AST NODE
# =========================================================
Node = tuple[str, object, object, Optional[float]]


def render(n: Node) -> str:
    if n[0] == "lit":
        return str(n[3])

    if n[0] == "powbase":
        return f"({BASE}**{n[1]})"

    if n[0] == "add":
        return f"({render(n[1])}+{render(n[2])})"

    if n[0] == "sub":
        return f"({render(n[1])}-{render(n[2])})"

    if n[0] == "mul":
        return f"({render(n[1])}*{render(n[2])})"

    if n[0] == "pow":
        return f"({render(n[1])}**{render(n[2])})"

    raise ValueError(n[0])


def evaluate_cost(expr: str) -> tuple[int, float]:
    """Deterministic evaluator used for final verification/debug."""

    def _eval(node):
        if isinstance(node, ast.BinOp):
            lv, lc = _eval(node.left)
            rv, rc = _eval(node.right)

            if isinstance(node.op, ast.Add):
                return lv + rv, lc + rc + COST["add"]
            if isinstance(node.op, ast.Sub):
                return lv - rv, lc + rc + COST["sub"]
            if isinstance(node.op, ast.Mult):
                return lv * rv, lc + rc + COST["mul"]
            if isinstance(node.op, ast.Pow):
                # Special-case BASE**n as powbase
                if (
                    isinstance(node.left, ast.Constant)
                    and node.left.value == BASE
                    and isinstance(node.right, ast.Constant)
                ):
                    return BASE**node.right.value, COST["powbase"]
                return lv**rv, lc + rc + COST["exp"]
            raise ValueError("Unsupported op")

        if isinstance(node, ast.Constant):
            return node.value, COST["lit"]

        raise ValueError("Unsupported AST node")

    t = ast.parse(expr, mode="eval")
    return _eval(t.body)


@lru_cache(maxsize=None)
def is_power(n, base):
    if n < 1 or base < 2:
        return False

    # Fast path for your actual BASE=2 case.
    if base == 2:
        return (n & (n - 1)) == 0

    while n % base == 0:
        n //= base
    return n == 1


# Precompute exact powers once. This is the big win.
_MAX_PRECOMP_VAL = (1 << 31) - 1
if BASE == 2:
    _MAX_BASE_EXP = _MAX_PRECOMP_VAL.bit_length() - 1
    _BASE_POWERS_ALL = [1 << e for e in range(1, _MAX_BASE_EXP + 1)]
else:
    _MAX_BASE_EXP = int(math.log(_MAX_PRECOMP_VAL, BASE))
    _BASE_POWERS_ALL = [
        BASE**e for e in range(1, _MAX_BASE_EXP + 1) if BASE**e <= _MAX_PRECOMP_VAL
    ]

_EXACT_POWER_MAP: dict[int, list[tuple[int, int]]] = {}
_EXACT_POWER_LIST: list[tuple[int, int, int]] = []  # (value, base, exp)

for b in range(2, math.isqrt(_MAX_PRECOMP_VAL)):
    p = b * b
    e = 2
    while p <= _MAX_PRECOMP_VAL:
        _EXACT_POWER_MAP.setdefault(p, []).append((b, e))
        _EXACT_POWER_LIST.append((p, b, e))
        e += 1
        p *= b

_EXACT_POWER_LIST.sort(key=lambda t: t[0])
_EXACT_POWER_VALUES = [x[0] for x in _EXACT_POWER_LIST]


def build_balanced(nodes):
    level = list(nodes)
    while len(level) > 1:
        nxt = []
        it = iter(level)
        for left in it:
            try:
                right = next(it)
            except StopIteration:
                nxt.append(left)
            else:
                nxt.append(("add", left, right, None))
        level = nxt
    return level[0] if level else None


def fold_pow(node):
    # Only fold a ** b when both are constants
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if (
            isinstance(node.left, ast.Constant)
            and node.left.value == BASE
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, int)
        ):
            return ast.Constant(value=node.left.value**node.right.value)
    return node


def flatten_add(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return flatten_add(node.left) + flatten_add(node.right)
    else:
        return [node]


def transform(node):
    # Recursively process children first
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            setattr(node, field, transform(value))
        elif isinstance(value, list):
            setattr(
                node,
                field,
                [transform(v) if isinstance(v, ast.AST) else v for v in value],
            )

    # Then apply local rewrite
    node = fold_pow(node)
    return node


def rebuild_add(terms):
    expr = terms[0]
    for term in terms[1:]:
        expr = ast.BinOp(left=expr, op=ast.Add(), right=term)
    return expr


def simplify_expr(expr_str):
    tree = ast.parse(expr_str, mode="eval")
    tree = transform(tree)

    terms = flatten_add(tree.body)
    new_tree = rebuild_add(terms)

    return ast.unparse(new_tree)


def synthesize_optimal_with_exp(
    target: int,
    disallowed: Optional[set[int]] = None,
    max_ext: int = 26,
    max_val: int = (1 << 31) - 1,
    verbose: bool = False,
) -> tuple[float, str]:

    disallowed = set() if disallowed is None else set(disallowed)
    nodes = 0

    # Keep base powers local to the current max_val.
    if BASE == 2:
        max_base_exp = max_val.bit_length() - 1
        base_powers = [1 << e for e in range(1, max_base_exp + 1)]
    else:
        max_base_exp = int(math.log(max_val, BASE))
        base_powers = [
            BASE**e for e in range(1, max_base_exp + 1) if BASE**e <= max_val
        ]

    @lru_cache(maxsize=None)
    def divisor_pairs(n: int) -> tuple[tuple[int, int], ...]:
        """Exact factor pairs only. Odd scan cuts half the work immediately."""
        if n < 4:
            return tuple()

        out = []
        if n % 2 == 0:
            out.append((2, n // 2))

        r = math.isqrt(n)
        a = 3
        while a <= r:
            if n % a == 0:
                out.append((a, n // a))
            a += 2

        return tuple(out)

    @lru_cache(maxsize=None)
    def simple_synthesis(value: int) -> tuple[float, Node]:
        if value <= max_ext and value not in disallowed:
            return COST["lit"], ("lit", None, None, value)

        if value > 0:
            if BASE == 2:
                if value & (value - 1) == 0:
                    n = value.bit_length() - 1
                    if n not in disallowed:
                        return COST["powbase"], ("powbase", n, None, None)
                else:
                    n = int(round(math.log(value, BASE)))
                    if BASE**n == value and n not in disallowed:
                        return COST["powbase"], ("powbase", n, None, None)

        digits = []
        temp = value
        power = 0
        while temp > 0:
            digit = temp % BASE
            if digit != 0:
                digits.append((digit, power))
            temp //= BASE
            power += 1

        if not digits:
            return COST["lit"], ("lit", None, None, 0)

        # digits.sort(key=lambda x: x[1])
        parts = []
        total_cost = 0.0
        i = 0

        while i < len(digits):
            digit, start = digits[i]

            if digit == 1:
                j = i
                while (
                    j + 1 < len(digits)
                    and digits[j + 1][0] == 1
                    and digits[j + 1][1] == digits[j][1] + 1
                ):
                    j += 1

                end = digits[j][1]
                length = end - start + 1

                if length >= 3:
                    streak_parts = [
                        ("powbase", p, None, None) for p in range(start, end + 1)
                    ]
                    parts.append(build_balanced(streak_parts))
                    total_cost += length * COST["powbase"] + (length - 1) * COST["add"]
                else:
                    for p in range(start, end + 1):
                        parts.append(("powbase", p, None, None))
                        total_cost += COST["powbase"]

                i = j + 1
            else:
                if start == 0:
                    parts.append(("lit", None, None, digit))
                    total_cost += COST["lit"]
                else:
                    parts.append(
                        (
                            "mul",
                            ("lit", None, None, digit),
                            ("powbase", start, None, None),
                            None,
                        )
                    )
                    total_cost += COST["powbase"] + COST["mul"] + COST["lit"]
                i += 1

        expr = build_balanced(parts)
        if len(parts) > 1:
            total_cost += (len(parts) - 1) * COST["add"]

        return total_cost, expr

    @lru_cache(maxsize=None)
    def near_exact_powers(value: int) -> tuple[tuple[int, int, int], ...]:
        """
        Only exact powers within +/- max_ext can matter for:
        (a**b) +/- k, so don't scan the entire table every time.
        """
        lo = max(0, value - max_ext)
        hi = min(max_val, value + max_ext)
        left = bisect.bisect_left(_EXACT_POWER_VALUES, lo)
        right = bisect.bisect_right(_EXACT_POWER_VALUES, hi)
        return tuple(_EXACT_POWER_LIST[left:right])

    @lru_cache(maxsize=None)
    def candidate_add_splits(value: int) -> tuple[int, ...]:
        """
        Structural pruning, not best-K selection.
        Small, midpoint-biased, and power-aware only.
        """
        if value <= 1:
            return tuple()

        cand = set()

        # Small seeds. Enough to catch a lot, without turning into a haystack.
        seeds = (1, 2, 3, 4, 5, 6, 7, 8, 13, 21, max_ext)
        for x in seeds:
            if 0 < x < value:
                cand.add(x)
                cand.add(value - x)

        # Powers of BASE and their complements.
        idx = bisect.bisect_left(base_powers, value)
        for i in (idx - 3, idx - 2, idx - 1, idx, idx + 1):
            if 0 <= i < len(base_powers):
                p = base_powers[i]
                if p < value:
                    cand.add(p)
                    cand.add(value - p)
        # Midpoint bias, because humans and search both love symmetry.
        half = value // 2
        for delta in (0, 1, 2, 3, 5, 8, 13):
            for x in (half - delta, half + delta):
                if 1 < x < value:
                    cand.add(x)

        return tuple(sorted(cand))

    depth_limit = 3

    @lru_cache(maxsize=None)
    def solve(value: int, depth: int = 0) -> tuple[float, Node]:
        nonlocal nodes
        nodes += 1
        if value not in disallowed:
            if value <= max_ext:
                return COST["lit"], ("lit", None, None, value)

            if is_power(value, BASE):
                if BASE == 2:
                    n = value.bit_length() - 1
                else:
                    n = int(round(math.log(value, BASE)))
                if n not in disallowed:
                    return COST["powbase"], ("powbase", n, None, None)

        if depth > depth_limit:
            return simple_synthesis(value)

        best_cost, best_expr = simple_synthesis(value)

        # BASE^n +/- k, but only when k is small.
        for anchor in base_powers:
            if anchor == value:
                continue

            if anchor > value:
                k = anchor - value
                if k <= max_ext and k not in disallowed:
                    cost_k, expr_k = solve(k, depth + 1)
                    total = COST["powbase"] + cost_k + COST["sub"]
                    if total < best_cost:
                        if BASE == 2:
                            n = anchor.bit_length() - 1
                        else:
                            n = int(round(math.log(anchor, BASE)))
                        best_cost = total
                        best_expr = ("sub", ("powbase", n, None, None), expr_k, None)
            else:
                k = value - anchor
                if k <= max_ext and k not in disallowed:
                    cost_k, expr_k = solve(k, depth + 1)
                    total = COST["powbase"] + cost_k + COST["add"]
                    if total < best_cost:
                        if BASE == 2:
                            n = anchor.bit_length() - 1
                        else:
                            n = int(round(math.log(anchor, BASE)))
                        best_cost = total
                        best_expr = ("add", ("powbase", n, None, None), expr_k, None)

        # Exact powers near the target only.
        for p, base, exp in near_exact_powers(value):
            if p == value:
                continue
            if base in disallowed or exp in disallowed:
                continue

            k = abs(p - value)
            if k <= max_ext and k not in disallowed:
                cost_base, expr_base = solve(base, depth + 1)
                cost_exp, expr_exp = solve(exp, depth + 1)
                cost_k, expr_k = solve(k, depth + 1)

                total = cost_base + cost_exp + cost_k + COST["exp"]
                total += COST["sub"] if p > value else COST["add"]

                if total < best_cost:
                    op = "sub" if p > value else "add"
                    best_cost = total
                    best_expr = (op, ("pow", expr_base, expr_exp, None), expr_k, None)

        # Exact powers.
        for base, exp in _EXACT_POWER_MAP.get(value, ()):
            if base in disallowed or exp in disallowed:
                continue

            cost_base, expr_base = solve(base, depth + 1)
            cost_exp, expr_exp = solve(exp, depth + 1)
            total = cost_base + cost_exp + COST["exp"]

            if total < best_cost:
                best_cost = total
                best_expr = ("pow", expr_base, expr_exp, None)

        # Exact factor pairs.
        for a, b in divisor_pairs(value):
            if a in disallowed or b in disallowed:
                continue

            cost_a, expr_a = solve(a, depth + 1)
            cost_b, expr_b = solve(b, depth + 1)
            total = cost_a + cost_b + COST["mul"]

            if total < best_cost:
                best_cost = total
                best_expr = ("mul", expr_a, expr_b, None)

        # Reduced additive search.
        for a in candidate_add_splits(value):
            b = value - a
            if a in disallowed or b in disallowed:
                continue

            cost_a, expr_a = solve(a, depth + 1)
            cost_b, expr_b = solve(b, depth + 1)
            total = cost_a + cost_b + COST["add"]

            if total < best_cost:
                best_cost = total
                best_expr = ("add", expr_a, expr_b, None)

            # Only try factor-sharing on promising additive splits.
            if a > 1 and b > 1:
                up = min(20, math.isqrt(min(a, b)))
                for f in range(2, up + 1):
                    if a % f == 0 and b % f == 0:
                        a2 = a // f
                        b2 = b // f
                        cost_f, expr_f = solve(f, depth + 1)
                        cost_a2, expr_a2 = solve(a2, depth + 1)
                        cost_b2, expr_b2 = solve(b2, depth + 1)
                        total2 = cost_f + cost_a2 + cost_b2 + COST["mul"] + COST["add"]
                        if total2 < best_cost:
                            best_cost = total2
                            best_expr = (
                                "mul",
                                expr_f,
                                ("add", expr_a2, expr_b2, None),
                                None,
                            )
        return best_cost, best_expr

    cost, _ast = solve(target)

    if verbose:
        print(f"nodes: {nodes}")
    expr = simplify_expr(render(_ast))
    return cost, expr


def optimize_sum_with_U(target: list[int], U: int) -> str:
    total_target = sum(target)
    return synthesize_optimal_with_exp(
        total_target,
        disallowed=set(),
        max_ext=U,
        max_val=(1 << 31) - 1,
        verbose=False,
    )[1]


optimizations = [optimize_sum_with_U]

if __name__ == "__main__":
    import time

    test_cases = [
        (56, set(), 26),
        (100, set(), 26),
        (22899, {10}, 26),
        (1000, set(), 26),
        (123, set(), 26),
        (319216, set(), 26),
        (2**28 + 2**29, set(), 27),
        (1073741824, set(), 26),  # 2^30
        (1073741825, set(), 26),  # 2^30 + 1
        (2147483647, set(), 26),  # 2^31 - 1
        (1048575, set(), 26),  # 2^20 - 1
        (654321, set(), 26),  # Has contiguous sequences
        (999999, set(), 26),  # Another test
        (4953, {10}, 26),
        (4983, {10}, 26),
        (36, set(), 26),  # 6**2
        (64, set(), 26),  # 4**3 or 8**2
        (125, set(), 26),  # 5**3
        (216, set(), 26),  # 6**3
        (6896, set(), 26),
        (3955, set(), 26),
        (166375, set(), 55),
    ]

    print("Testing synthesis with exponentiation patterns...")
    print("=" * 60)
    print(f"BASE: {BASE}")
    print(f"Cost configuration: {COST}")

    for target, disallowed, max_ext in test_cases:
        print(f"\nTarget: {target}")
        if disallowed:
            print(f"Disallowed: {disallowed}")

        start = time.time()
        cost, expr = synthesize_optimal_with_exp(
            target,
            disallowed=disallowed,
            max_ext=max_ext,
            max_val=(1 << 31) - 1,
            verbose=False,
        )
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.6f}s")
        print(f"  Cost: {cost:.2f}")
        print(f"  Expression: {expr}")

        # Verify
        actual_value, actual_cost = evaluate_cost(expr)
        if actual_value == target:
            print(f"  ✓ Verified (actual cost: {actual_cost:.2f})")

            # Show alternative possibilities for comparison
            if target == 319216:
                print(
                    f"  Alternative: ((2**4) * (((2**6)+(2**3))-1) * ((2**8)+(5**2))) = cost {evaluate_cost('((2**4) * (((2**6)+(2**3))-1) * ((2**8)+(5**2)))')[1]:.3f}"
                )
            elif target == 6896:
                print(
                    f"  Alternative: (2**4 * (2**9 - 3**4)) = cost {evaluate_cost('2**4 * (2**9 - 3**4)')[1]:.3f}"
                )
            elif target == 3955:
                print(
                    f"  Alternative: 19 * 26 * 8 + 3 = cost {evaluate_cost('19 * 26 * 8 + 3')[1]:.3f}"
                )
        else:
            print(f"  ✗ Wrong value: {actual_value} != {target}")
