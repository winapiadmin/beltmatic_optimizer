from __future__ import annotations
import math
import ast
import bisect
from typing import Optional
from functools import cache

BASE = 2
COST = {
    "add": 1 / 2,
    "sub": 1 / 1,
    "mul": 1 / 1,
    "pow": 1 / 1.1,
    "powbase": 0.00,
    "lit": 0.00,
}

# (operation, a, b, value)
# powbase: BASE**value
# add, sub, mul, pow: a (operation) b
# lit: value
Node = tuple[str, tuple | int | None, tuple | int | None, Optional[int]]

def synthesize_optimal_with_exp(
    target: int,
    disallowed: Optional[set[int]] = None,
    max_ext: int = 26,
    max_val: int = (1 << 31) - 1,
    verbose: bool = False,
) -> tuple[float, str]:
    """
    Synthesizes an optimal arithmetic expression to compute the given target integer using a set of allowed operations,
    minimizing a cost function based on operation types and optionally avoiding certain intermediate values.

    Parameters:
        target (int): The integer value to synthesize an expression for.
        disallowed (Optional[set[int]]): A set of integer values that must not appear as intermediate results.
        max_ext (int): The maximum value for which literals are allowed in the expression.
        max_val (int): The maximum value allowed for any intermediate computation.
        verbose (bool): If True, prints diagnostic information during synthesis.

    Returns:
        tuple[float, str]: A tuple containing the minimal cost and the corresponding arithmetic expression as a string.
    """

    disallowed = disallowed or set()
    nodes = 0

    # Keep base powers local to the current max_val.
    max_base_exp = int(math.log(max_val, BASE))
    base_powers = tuple(_BASE_POWERS_ALL[:bisect.bisect_right(_BASE_POWERS_ALL, max_val)])

    depth_limit = 4
    cost_add = COST["add"]
    cost_sub = COST["sub"]
    cost_mul = COST["mul"]
    cost_pow = COST["pow"]
    cost_powbase = COST["powbase"]
    cost_lit = COST["lit"]

    @cache
    def simple_synthesis(value: int) -> tuple[float, Node|None]:
        if value <= max_ext and value not in disallowed:
            return cost_lit, ("lit", None, None, value)

        if value > 0:
            if BASE == 2:
                if value & (value - 1) == 0:
                    n = value.bit_length() - 1
                    return cost_powbase, ("powbase", n, None, None)
            else:
                n = int(round(math.log(value, BASE)))
                if BASE**n == value:
                    return cost_powbase, ("powbase", n, None, None)

        digits = []
        temp = value
        power = 0
        while temp:
            temp, digit = divmod(temp, BASE)
            if digit:
                digits.append((digit, power))
            power += 1
        n_digits = len(digits)

        if not digits:
            return cost_lit, ("lit", None, None, 0)

        parts = []
        total_cost = 0.0
        i = 0

        while i < n_digits:
            digit, start = digits[i]

            if digit == 1:
                j = i
                while (
                    j + 1 < n_digits
                    and digits[j + 1][0] == 1
                    and digits[j + 1][1] == digits[j][1] + 1
                ):
                    j += 1

                end = digits[j][1]
                length = end - start + 1

                if length >= 3:
                    cost_uncompressed=length*cost_powbase + (length - 1) * cost_add
                    cost_compressed=COST["sub"]+2*COST["powbase"]
                    if cost_uncompressed<cost_compressed or BASE**(end+1)>=max_val or BASE!=2:
                        streak_parts = []

                        for p in range(start, end + 1):
                            streak_parts.append(("powbase", p, None, None))

                        parts.append(build_balanced(streak_parts))
                        total_cost += cost_uncompressed
                    else:
                        parts.append(("sub", ("powbase", end + 1, None, None), ("powbase", start, None, None), None))
                        total_cost += cost_compressed
                else:
                    for p in range(start, end + 1):
                        parts.append(("powbase", p, None, None))
                        total_cost += cost_powbase
                i = j + 1
            else:
                if start == 0:
                    parts.append(("lit", None, None, digit))
                    total_cost += cost_lit
                else:
                    parts.append(
                        (
                            "mul",
                            ("lit", None, None, digit),
                            ("powbase", start, None, None),
                            None,
                        )
                    )
                    total_cost += cost_powbase + cost_mul + cost_lit
                i += 1

        expr = build_balanced(parts)
        if len(parts) > 1:
            total_cost += (len(parts) - 1) * cost_add

        return total_cost, expr

    @cache
    def solve(value: int, depth: int = depth_limit) -> tuple[float, Node|None]:
        nonlocal nodes
        nodes += 1
        if value not in disallowed:
            if value <= max_ext:
                return cost_lit, ("lit", None, None, value)

            if is_power(value, BASE):
                if BASE == 2:
                    n = value.bit_length() - 1
                else:
                    n = int(round(math.log(value, BASE)))
                if n not in disallowed:
                    return cost_powbase, ("powbase", n, None, None)

        if depth <= 0:
            return simple_synthesis(value)

        best_cost, best_expr = simple_synthesis(value)

        def consider(cost: float, expr: Node|None) -> None:
            nonlocal best_cost, best_expr
            if cost >= best_cost - 1e-12:
                return
            best_cost = cost
            best_expr = expr

        if cost_powbase < best_cost:
            for n, anchor in enumerate(base_powers, start=1):
                if anchor == value:
                    continue
                if n in disallowed:
                    continue
                for k in range(1, 32):
                    if k in disallowed:
                        continue
                    if anchor * k > max_val:
                        break

                    cost_k, expr_k = solve(k, depth - 1)
                    delta = value - anchor * k
                    r = abs(delta)
                    if r <= max_ext * 2 and r not in disallowed:
                        op_cost = cost_sub if delta < 0 else cost_add
                        total = cost_powbase + op_cost * (r != 0) + (cost_mul + cost_k) * (k != 1)
                        if total >= best_cost:
                            continue
                        cost_r, expr_r = solve(r, depth - 1)
                        total += cost_r * (r != 0)
                        entry = ("powbase", n, None, None)
                        if k != 1:
                            entry = ("mul", expr_k, entry, None)
                        if r != 0:
                            entry = ("sub", entry, expr_r, None) if delta < 0 else ("add", entry, expr_r, None)
                        consider(total, entry)

        if cost_pow < best_cost:
            for p, base, exp in near_exact_powers(value, max_ext, max_val):
                if base in disallowed or exp in disallowed:
                    continue

                k = abs(p - value)
                if k <= max_ext * 2 and k not in disallowed:
                    cost_base, expr_base = solve(base, depth - 1)
                    if cost_base + cost_pow >= best_cost:
                        continue
                    cost_exp, expr_exp = solve(exp, depth - 1)
                    op_cost = cost_sub if p > value else cost_add
                    total = cost_base + cost_exp + cost_pow + op_cost * (k != 0)
                    if total >= best_cost:
                        continue
                    cost_k, expr_k = solve(k, depth - 1)
                    total += cost_k * (k != 0)
                    entry = ("pow", expr_base, expr_exp, None)
                    if k != 0:
                        entry = ("sub", entry, expr_k, None) if p > value else ("add", entry, expr_k, None)
                    consider(total, entry)

        if cost_mul < best_cost:
            for a, b in divisor_pairs(value):
                if a in disallowed or b in disallowed or a==1 or b==1: # useless values that leads to the same value
                    continue

                cost_a, expr_a = solve(a, depth - 1)
                if cost_a + cost_mul >= best_cost:
                    continue
                cost_b, expr_b = solve(b, depth - 1)
                total = cost_a + cost_b + cost_mul
                consider(total, ("mul", expr_a, expr_b, None))

        if cost_add < best_cost:
            for a in candidate_add_splits(value, max_ext, base_powers):
                b = value - a
                if a in disallowed or b in disallowed or a == 0 or b == 0:
                    continue

                cost_a, expr_a = solve(a, depth - 1)
                if cost_a + cost_add < best_cost:
                    cost_b, expr_b = solve(b, depth - 1)
                    total = cost_a + cost_b + cost_add
                    consider(total, ("add", expr_a, expr_b, None))
                if cost_mul + cost_add < best_cost:
                    if a > 1 and b > 1:
                        for f, g in divisor_pairs(math.gcd(a, b)):
                            for factor in (f, g):
                                if factor <= 1:
                                    continue

                                a2 = a // factor
                                b2 = b // factor
                                cost_f, expr_f = solve(factor, depth - 1)
                                if cost_f + cost_mul + cost_add >= best_cost:
                                    continue
                                cost_a2, expr_a2 = solve(a2, depth - 1)
                                if cost_f + cost_a2 + cost_mul + cost_add >= best_cost:
                                    continue
                                cost_b2, expr_b2 = solve(b2, depth - 1)
                                total2 = cost_f + cost_a2 + cost_b2 + cost_mul + cost_add
                                consider(
                                    total2,
                                    ("mul", expr_f, ("add", expr_a2, expr_b2, None), None),
                                )

        return best_cost, best_expr

    simple_synthesis.cache_clear()
    solve.cache_clear()
    if verbose:
        print("near_exact_powers:",near_exact_powers(target, max_ext, max_val))
        print("candidate_add_splits:", candidate_add_splits(target,max_ext, base_powers))
        cost, expr=simple_synthesis(target)
        print("simple_synthesis:", (cost, render(expr) if expr else None))
    cost, expr_tree = solve(target, depth_limit)

    if verbose:
        print(f"nodes: {nodes}")
    if expr_tree is not None:
        # Pipeline: tuple-based AST (expr_tree) -> string expression (render) -> simplified string (simplify_expr)
        expr = render(expr_tree)
    else: expr = "None"
    return cost, expr


def optimize_sum_with_U(target: list[int], U: int) -> str:
    total_target = sum(target)
    return simplify_expr(synthesize_optimal_with_exp(
        total_target,
        disallowed=set(),
        max_ext=U,
        max_val=(1 << 31) - 1,
        verbose=False,
    )[1])


optimizations = [optimize_sum_with_U]


def render(n: Node) -> str:
    if n[0] == "lit":
        return str(n[3])

    if n[0] == "powbase":
        return f"{BASE}**{n[1]}"

    if n[0] == "add":
        assert isinstance(n[1], tuple)
        assert isinstance(n[2], tuple)
        return f"({render(n[1])}+{render(n[2])})"

    if n[0] == "sub":
        assert isinstance(n[1], tuple)
        assert isinstance(n[2], tuple)
        return f"({render(n[1])}-{render(n[2])})"

    if n[0] == "mul":
        assert isinstance(n[1], tuple)
        assert isinstance(n[2], tuple)
        return f"{render(n[1])}*{render(n[2])}"

    if n[0] == "pow":
        assert isinstance(n[1], tuple)
        assert isinstance(n[2], tuple)
        return f"{render(n[1])}**{render(n[2])}"

    raise ValueError(n[0])


def evaluate_cost(expr: str) -> tuple[int, float]:
    """Deterministic evaluator used for final verification/debug."""

    def _eval(node: ast.AST) -> tuple[int, float]:
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
                    and isinstance(node.right.value, int)
                ):
                    return BASE**node.right.value, COST["powbase"]
                return lv**rv, lc + rc + COST["pow"]
            raise ValueError(f"Unsupported op: {type(node.op).__name__}\n{ast.dump(node, indent=2)}")

        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value, COST["lit"]

        raise ValueError("Unsupported AST node")

    t = ast.parse(expr, mode="eval")
    return _eval(t.body)


@cache
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
_MAX_BASE_EXP = int(math.log(_MAX_PRECOMP_VAL, BASE))
_BASE_POWERS_ALL = [
    BASE**e for e in range(1, _MAX_BASE_EXP + 1) if BASE**e <= _MAX_PRECOMP_VAL
]

_EXACT_POWER_MAP: dict[int, list[tuple[int, int]]] = {}
_EXACT_POWER_LIST: list[tuple[int, int, int]] = []  # (value, base, exp)

for b in range(2, math.isqrt(_MAX_PRECOMP_VAL) + 1):
    p = b * b
    e = 2
    while p <= _MAX_PRECOMP_VAL:
        _EXACT_POWER_MAP.setdefault(p, []).append((b, e))
        _EXACT_POWER_LIST.append((p, b, e))
        e += 1
        p *= b

_EXACT_POWER_LIST.sort(key=lambda t: t[0])
_EXACT_POWER_VALUES = [x[0] for x in _EXACT_POWER_LIST]
def build_balanced(nodes: list) -> Node | None:
    n = len(nodes)

    if n == 0:
        return None

    while n > 1:
        write = 0
        read = 0

        while read < n:
            left = nodes[read]
            read += 1

            if read < n:
                nodes[write] = ("add", left, nodes[read], None)
                read += 1
            else:
                nodes[write] = left

            write += 1

        n = write

    return nodes[0]
class Simplifier(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        # First recurse properly
        new_node = self.generic_visit(node)

        # ---- fold_pow ----
        if isinstance(new_node, ast.BinOp) and isinstance(new_node.op, ast.Pow):
            if (
                isinstance(new_node.left, ast.Constant)
                and new_node.left.value == BASE
                and isinstance(new_node.right, ast.Constant)
                and isinstance(new_node.right.value, int)
                and isinstance(new_node.left.value, int)
            ):
                return ast.Constant(value=new_node.left.value ** new_node.right.value)

        return new_node


class AddFlattener(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)

        if not isinstance(node.op, ast.Add):
            return node

        terms = []

        def collect(n):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                collect(n.left)
                collect(n.right)
            else:
                terms.append(n)

        collect(node)

        expr = terms[0]
        for term in terms[1:]:
            expr = ast.BinOp(left=expr, op=ast.Add(), right=term)

        return expr


class ExpressionSimplifier:
    def __init__(self):
        self.transformer = Simplifier()
        self.flattener = AddFlattener()

    def simplify(self, expr_str: str) -> str:
        tree = ast.parse(expr_str, mode="eval")

        tree = self.transformer.visit(tree)
        tree = self.flattener.visit(tree)

        ast.fix_missing_locations(tree)

        return ast.unparse(tree)

@cache
def divisor_pairs(n: int):
    out = [(1,n)]
    for a in range(2, math.isqrt(n) + 1):
        if n % a == 0:
            out.append((a, n // a))
    return tuple(out)

@cache
def near_exact_powers(value: int, max_ext: int, max_val:int) -> tuple[tuple[int, int, int], ...]:
    r = max_ext
    lo = max(0, value - r)
    hi = min(max_val, value + r)
    left = bisect.bisect_left(_EXACT_POWER_VALUES, lo)
    right = bisect.bisect_right(_EXACT_POWER_VALUES, hi)
    return tuple(_EXACT_POWER_LIST[left:right])

@cache
def candidate_add_splits(value: int, max_ext: int, base_powers:tuple[int]) -> tuple[int, ...]:
    if value <= 1:
        return tuple()
    cand = set()
    max_test = max(max_ext, 32)*2

    for x in range(1, max_test + 1):
        if x <= value:
            cand.add(x)
            cand.add(value - x)

    idx = bisect.bisect_left(base_powers, value)
    for i_ in range(-5,+2):
        i=idx+i_
        if 0 <= i < len(base_powers):
            p = base_powers[i]
            if p <= value:
                cand.add(p)
                cand.add(value - p)

    for i in range(max(0, idx - 2), min(len(base_powers), idx + 3)):
        p = base_powers[i]
        if p <= 1:
            continue
        max_k = min(32, (value - 1) // p)
        for k in range(2, max_k + 1):
            x = k * p
            if x < value:
                cand.add(x)
                cand.add(value - x)

    half = value // 2
    for delta in (0, 1, 2, 3, 5, 8, 13):
        for x in (half - delta, half + delta):
            if 1 < x < value:
                cand.add(x)

    return tuple(sorted(cand))
simplify_expr=ExpressionSimplifier().simplify

if __name__ == "__main__":
    import random
    def random_tc():
        return random.randrange(1,math.isqrt(2**31-1)), set(), 26
    print(random_tc())
    import time
    test_cases = [
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
        (1368794382, set(), 27),
        (222860571, set(), 26),
        (132893794, set(), 26),
        (1150000477, set(), 26),
        (295052544, set(), 27),
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
            max_val=2**31-1,
            verbose=True,
        )
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.6f}s")
        print(f"  Cost: {cost:.2f}")
        print(f"  Expression: {expr} (simplified: {simplify_expr(expr)})")

        # Verify
        actual_value, actual_cost = evaluate_cost(expr)

        # Display alternatives
        def alt(equation: str):
            assert evaluate_cost(equation)[0] == target
            print(
                f"  Alternative: {equation} = cost {evaluate_cost(equation)[1]:.3f}"
            )

        if actual_value == target:
            if actual_cost != cost:
                print(f"  ✗ Wrong cost (actual cost: {actual_cost:.2f}")
            else: 
                print(f"  ✓ Verified (actual cost: {actual_cost:.2f})")

            # Show alternative possibilities for comparison
            match target:
                case 319216:
                    alt("2**16 + 15 * (16 + 2**9 + 2**14)")
                case 6896:
                    alt("(2**6 + 19) ** 2 + 7")
                case 3955:
                    alt("5 * (2**8 + 2**9 + 23)")
                case 1368794382:
                    alt("2**28 + (2**10 + 15) * (25 * 2**10 + 18) + 2**30")
                    alt("14 + (14 + 2**4 + 2**10)*6**7 + 2**30")
                case 654321:
                    alt("3*(2**13+2**15+3**11)")
                case 989:
                    alt("2 ** 10 - (2**5 + 3)")
        else:
            print(f"  ✗ Wrong value: {actual_value} != {target}")
