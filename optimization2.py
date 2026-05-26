from __future__ import annotations
import heapq
from bisect import bisect_left, bisect_right
from math import gcd, isqrt, log
from dataclasses import dataclass
from functools import cache

import hashlib
import json
import pprint
import random
import time
from pathlib import Path

BASE = 2
COST = {
    "add": 1 / 2,
    "sub": 1 / 1,
    "mul": 1 / 1,
    "pow": 1 / 1.1,
    "powbase": 0.00,
    "lit": 0.00,
}


@dataclass(frozen=True, slots=True)
class LitNode:
    value: int


@dataclass(frozen=True, slots=True)
class PowBaseNode:
    exponent: int


@dataclass(frozen=True, slots=True)
class BinOpNode:
    op: str
    left: Node
    right: Node


Node = LitNode | PowBaseNode | BinOpNode


def _tokenize(s: str) -> list:
    tokens: list = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in "()":
            tokens.append(c)
            i += 1
        elif c == "+":
            tokens.append("+")
            i += 1
        elif c == "-":
            tokens.append("-")
            i += 1
        elif c == "*":
            if i + 1 < len(s) and s[i + 1] == "*":
                tokens.append("**")
                i += 2
            else:
                tokens.append("*")
                i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(int(s[i:j]))
            i = j
        else:
            i += 1
    return tokens


def _parse_primary(tokens: list, pos: int) -> tuple[Node, int]:
    """Parse primary: '(' expr ')' | int | BASE**int"""
    tok = tokens[pos]
    if tok == "(":
        pos += 1
        node, pos = _parse_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError(f"Expected ')' at position {pos}")
        pos += 1
        return node, pos
    if isinstance(tok, int):
        val = tok
        pos += 1
        if val == BASE and pos < len(tokens) and tokens[pos] == "**":
            pos += 1
            if pos < len(tokens) and isinstance(tokens[pos], int):
                k = tokens[pos]
                pos += 1
                return PowBaseNode(k), pos
            right, pos = _parse_primary(tokens, pos)
            return BinOpNode("pow", LitNode(BASE), right), pos
        return LitNode(val), pos
    raise ValueError(f"Unexpected token: {tok}")


def _parse_muldiv(tokens: list, pos: int) -> tuple[Node, int]:
    """Parse muldiv: primary (('*' | '**') primary)*"""
    left, pos = _parse_primary(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("*", "**"):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_primary(tokens, pos)
        if op == "**":
            if (
                isinstance(left, LitNode)
                and left.value == BASE
                and isinstance(right, LitNode)
            ):
                left = PowBaseNode(right.value)
            else:
                left = BinOpNode("pow", left, right)
        else:
            left = BinOpNode("mul", left, right)
    return left, pos


def _parse_expr(tokens: list, pos: int) -> tuple[Node, int]:
    """Parse addsub_expr: muldiv (('+' | '-') muldiv)*"""
    left, pos = _parse_muldiv(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_muldiv(tokens, pos)
        if op == "+":
            left = BinOpNode("add", left, right)
        else:
            left = BinOpNode("sub", left, right)
    return left, pos


def parse_expr(s: str) -> Node:
    tokens = _tokenize(s)
    node, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unexpected trailing tokens: {tokens[pos:]}")
    return node


# ======================================================
# BFS-on-cost optimal table (moved from optimization3.py)
# ======================================================

_DP_LIMIT = 50_000


@dataclass
class OptimalTable:
    cost: list[float] | None = None
    expr: list | None = None
    config_hash: int | None = None

    def ensure(
        self,
        max_literal: int = 26,
        max_value: int = (1 << 31) - 1,
    ) -> None:
        """Build optimal-cost table for all values ≤ _DP_LIMIT using BFS-on-cost."""
        h = hash((BASE, tuple(sorted(COST.items())), max_literal, max_value))
        if self.cost is not None and self.config_hash == h:
            return

        L = _DP_LIMIT
        limit = min(L, max_value)
        INF = float("inf")
        cost_arr: list[float] = [INF] * (L + 1)
        expr_arr: list = [None] * (L + 1)

        seeds: set[int] = set(range(max_literal + 1))
        p = BASE
        while p <= limit:
            seeds.add(p)
            p *= BASE

        for s in seeds:
            cost_arr[s] = 0.0
            if s <= max_literal:
                expr_arr[s] = LitNode(s)
            else:
                n = s.bit_length() - 1
                expr_arr[s] = PowBaseNode(n)

        pq: list[tuple[float, int]] = [(0.0, s) for s in seeds]
        heapq.heapify(pq)

        seed_list = sorted(seeds)
        small_non_seeds: list[int] = []

        while pq:
            cur_cost, val = heapq.heappop(pq)
            if cur_cost > cost_arr[val]:
                continue

            for seed in seed_list:
                seed_cost = cost_arr[seed]

                new_val = val + seed
                if new_val <= limit:
                    new_cost = cur_cost + seed_cost + COST["add"]
                    if new_cost < cost_arr[new_val]:
                        cost_arr[new_val] = new_cost
                        expr_arr[new_val] = BinOpNode(
                            "add", expr_arr[val], expr_arr[seed]
                        )
                        heapq.heappush(pq, (new_cost, new_val))

                if val >= seed:
                    new_val = val - seed
                    new_cost = cur_cost + seed_cost + COST["sub"]
                    if new_cost < cost_arr[new_val]:
                        cost_arr[new_val] = new_cost
                        expr_arr[new_val] = BinOpNode(
                            "sub", expr_arr[val], expr_arr[seed]
                        )
                        heapq.heappush(pq, (new_cost, new_val))

                if seed >= val:
                    new_val = seed - val
                    new_cost = cur_cost + seed_cost + COST["sub"]
                    if new_cost < cost_arr[new_val]:
                        cost_arr[new_val] = new_cost
                        expr_arr[new_val] = BinOpNode(
                            "sub", expr_arr[seed], expr_arr[val]
                        )
                        heapq.heappush(pq, (new_cost, new_val))

                new_val = val * seed
                if new_val <= limit and new_val != 0:
                    new_cost = cur_cost + seed_cost + COST["mul"]
                    if new_cost < cost_arr[new_val]:
                        cost_arr[new_val] = new_cost
                        expr_arr[new_val] = BinOpNode(
                            "mul", expr_arr[val], expr_arr[seed]
                        )
                        heapq.heappush(pq, (new_cost, new_val))

            if val >= 2:
                for seed in seed_list:
                    if 2 <= seed <= 15:
                        try:
                            new_val = val**seed
                        except (OverflowError, ValueError):
                            continue
                        if new_val <= limit:
                            new_cost = cur_cost + cost_arr[seed] + COST["pow"]
                            if new_cost < cost_arr[new_val]:
                                expr_arr[new_val] = BinOpNode(
                                    "pow", expr_arr[val], expr_arr[seed]
                                )
                                cost_arr[new_val] = new_cost
                                heapq.heappush(pq, (new_cost, new_val))

            if 2 <= val <= 15:
                for seed in seed_list:
                    if seed >= 2:
                        try:
                            new_val = seed**val
                        except (OverflowError, ValueError):
                            continue
                        if new_val <= limit:
                            new_cost = cur_cost + cost_arr[seed] + COST["pow"]
                            if new_cost < cost_arr[new_val]:
                                expr_arr[new_val] = BinOpNode(
                                    "pow", expr_arr[seed], expr_arr[val]
                                )
                                cost_arr[new_val] = new_cost
                                heapq.heappush(pq, (new_cost, new_val))

            if val < 200 and val not in seeds:
                for other in small_non_seeds:
                    if other == val:
                        continue
                    new_val = val * other
                    if new_val <= limit:
                        new_cost = cur_cost + cost_arr[other] + COST["mul"]
                        if new_cost < cost_arr[new_val]:
                            expr_arr[new_val] = BinOpNode(
                                "mul", expr_arr[val], expr_arr[other]
                            )
                            cost_arr[new_val] = new_cost
                            heapq.heappush(pq, (new_cost, new_val))
                    new_val = val + other
                    if new_val <= limit:
                        new_cost = cur_cost + cost_arr[other] + COST["add"]
                        if new_cost < cost_arr[new_val]:
                            expr_arr[new_val] = BinOpNode(
                                "add", expr_arr[val], expr_arr[other]
                            )
                            cost_arr[new_val] = new_cost
                            heapq.heappush(pq, (new_cost, new_val))

            if val not in seeds and val < 200:
                small_non_seeds.append(val)

        self.cost = cost_arr
        self.expr = expr_arr
        self.config_hash = h


_OPTIMAL_TABLE = OptimalTable()


def ensure_optimal_table(
    max_literal: int = 26,
    max_value: int = (1 << 31) - 1,
) -> None:
    _OPTIMAL_TABLE.ensure(max_literal, max_value)


def optimal_lookup(value: int) -> tuple[float, Node] | None:
    """Return (cost, expr) from the optimal table, or None if unavailable."""
    if _OPTIMAL_TABLE.cost is not None and value <= _DP_LIMIT:
        c = _OPTIMAL_TABLE.cost[value]
        if c != float("inf"):
            return c, _OPTIMAL_TABLE.expr[value]
    return None


def synthesize_optimal_with_exp(
    target: int,
    disallowed: set[int] | None = None,
    max_ext: int = 26,
    max_val: int = (1 << 31) - 1,
    verbose: bool = False,
) -> tuple[float, str]:
    """
    @brief Synthesizes a low-cost arithmetic expression for an integer target.

    Searches for an arithmetic expression minimizing weighted operation cost
    using addition, subtraction, multiplication, exponentiation, and BASE powers.

    @param target
        Integer value to synthesize.

    @param disallowed
        Literal constants forbidden from appearing directly in the emitted expression.

    @param max_ext
        Maximum literal value allowed for direct literal synthesis.

    @param max_val
        Maximum permitted intermediate value.

    @param verbose
        Enables diagnostic logging.

    @return
        Tuple containing:
        - minimal estimated cost
        - rendered expression string

    @note
        The search is heuristic and depth-limited. Optimality is not globally guaranteed.

    @warning
        Costs are floating-point values and may accumulate rounding error.
    """

    disallowed = disallowed or set()
    nodes = 0
    dp_hits = 0

    # Keep base powers local to the current max_val.
    if max_val >= _MAX_PRECOMP_VAL:
        base_powers = _BASE_POWERS
    else:
        base_powers = tuple(_BASE_POWERS_ALL[: bisect_right(_BASE_POWERS_ALL, max_val)])

    depth_limit = 4
    cost_add = COST["add"]
    cost_sub = COST["sub"]
    cost_mul = COST["mul"]
    cost_pow = COST["pow"]
    cost_powbase = COST["powbase"]
    cost_lit = COST["lit"]
    ensure_optimal_table(max_ext, max_val)

    @cache
    def simple_synthesis(value: int) -> tuple[float, Node | None]:
        if value <= max_ext and value not in disallowed:
            return cost_lit, LitNode(value)

        if value > 0:
            if BASE == 2:
                if value & (value - 1) == 0:
                    n = value.bit_length() - 1
                    return cost_powbase, PowBaseNode(n)
            else:
                n = int(round(log(value, BASE)))
                if BASE**n == value:
                    return cost_powbase, PowBaseNode(n)
        digits = []
        temp = value
        power = 0
        while temp:
            if temp & 1:
                digits.append((1, power))
            temp >>= 1
            power += 1
        n_digits = len(digits)

        if not digits:
            return cost_lit, LitNode(0)

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
                    cost_uncompressed = length * cost_powbase + (length - 1) * cost_add
                    cost_compressed = cost_sub + 2 * cost_powbase
                    if (
                        cost_uncompressed < cost_compressed
                        or BASE ** (end + 1) >= max_val
                        or BASE != 2
                    ):
                        streak_parts = []

                        for p in range(start, end + 1):
                            streak_parts.append(PowBaseNode(p))

                        parts.append(build_balanced(streak_parts))
                        total_cost += cost_uncompressed
                    else:
                        parts.append(
                            BinOpNode("sub", PowBaseNode(end + 1), PowBaseNode(start))
                        )
                        total_cost += cost_compressed
                else:
                    for p in range(start, end + 1):
                        parts.append(PowBaseNode(p))
                        total_cost += cost_powbase
                i = j + 1
            else:
                if start == 0:
                    parts.append(LitNode(digit))
                    total_cost += cost_lit
                else:
                    parts.append(
                        (
                            "mul",
                            LitNode(digit),
                            PowBaseNode(start),
                            None,
                        )
                    )
                    total_cost += cost_powbase + cost_mul + cost_lit
                i += 1

        expr = build_balanced(parts)
        if len(parts) > 1:
            total_cost += (len(parts) - 1) * cost_add

        return total_cost, expr

    _solve_cache = {}

    def solve(value: int, depth: int = depth_limit) -> tuple[float, Node | None]:
        nonlocal nodes, dp_hits
        cached = _solve_cache.get(value)
        if cached is not None and cached[1] >= depth:
            return cached[0]
        if not disallowed:
            dp_result = optimal_lookup(value)
            if dp_result is not None:
                dp_hits += 1
                _solve_cache[value] = (dp_result, depth_limit)
                return dp_result
        nodes += 1
        if value not in disallowed:
            if value <= max_ext:
                result = (cost_lit, LitNode(value))
                _solve_cache[value] = (result, depth)
                return result

            if is_power(value, BASE):
                if BASE == 2:
                    n = value.bit_length() - 1
                else:
                    n = int(round(log(value, BASE)))
                result = (cost_powbase, PowBaseNode(n))
                _solve_cache[value] = (result, depth)
                return result

        if depth <= 0:
            result = simple_synthesis(value)
            _solve_cache[value] = (result, depth)
            return result

        best_cost, best_expr = simple_synthesis(value)

        if cost_powbase < best_cost:
            for n, anchor in enumerate(base_powers, start=1):
                if anchor == value:
                    continue
                lo = max(1, (value - max_ext * 2 + anchor - 1) // anchor)
                hi = min(31, (value + max_ext * 2) // anchor, max_val // anchor)
                for k in range(lo, hi + 1):
                    delta = value - anchor * k
                    r = abs(delta)
                    if r not in disallowed:
                        total = cost_powbase
                        if k != 1:
                            cost_k, expr_k = solve(k, depth - 1)
                            total += cost_mul + cost_k
                        if r:
                            total += cost_sub if delta < 0 else cost_add
                        if total >= best_cost:
                            continue
                        if r:
                            cost_r, expr_r = solve(r, depth - 1)
                            total += cost_r
                        if total >= best_cost - 1e-12:
                            continue
                        entry = PowBaseNode(n)
                        if k != 1:
                            entry = BinOpNode("mul", expr_k, entry)
                        if r:
                            entry = (
                                BinOpNode("sub", entry, expr_r)
                                if delta < 0
                                else BinOpNode("add", entry, expr_r)
                            )
                        best_cost = total
                        best_expr = entry

        if cost_pow < best_cost:
            pow_candidates = sorted(
                near_exact_powers(value, max_ext, max_val),
                key=lambda pe: simple_synthesis(pe[1])[0] + simple_synthesis(pe[2])[0],
            )
            for p, base, exp in pow_candidates:
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
                    if k:
                        cost_k, expr_k = solve(k, depth - 1)
                        total += cost_k
                    entry = BinOpNode("pow", expr_base, expr_exp)
                    if k:
                        entry = (
                            BinOpNode("sub", entry, expr_k)
                            if p > value
                            else BinOpNode("add", entry, expr_k)
                        )
                    if total < best_cost - 1e-12:
                        best_cost = total
                        best_expr = entry

        if cost_mul < best_cost:
            pairs = sorted(
                divisor_pairs(value),
                key=lambda ab: simple_synthesis(ab[0])[0] + simple_synthesis(ab[1])[0],
            )
            for a, b in pairs:
                if a == 1 or b == 1:
                    continue

                if simple_synthesis(a)[0] + cost_mul >= best_cost:
                    continue
                cost_a, expr_a = solve(a, depth - 1)
                if cost_a + cost_mul >= best_cost:
                    continue
                cost_b, expr_b = solve(b, depth - 1)
                total = cost_a + cost_b + cost_mul
                if total < best_cost - 1e-12:
                    best_cost = total
                    best_expr = BinOpNode("mul", expr_a, expr_b)

        if cost_add < best_cost:
            splits = sorted(
                candidate_add_splits(value, max_ext, base_powers),
                key=lambda x: simple_synthesis(x)[0] + simple_synthesis(value - x)[0],
            )
            for a in splits:
                b = value - a
                if a == 0 or b == 0:
                    continue

                if simple_synthesis(a)[0] + cost_add >= best_cost:
                    continue
                cost_a, expr_a = solve(a, depth - 1)
                if cost_a + cost_add < best_cost:
                    cost_b, expr_b = solve(b, depth - 1)
                    total = cost_a + cost_b + cost_add
                    if total < best_cost - 1e-12:
                        best_cost = total
                        best_expr = BinOpNode("add", expr_a, expr_b)
                if cost_mul + cost_add < best_cost:
                    if a > 1 and b > 1:
                        g = gcd(a, b)
                        if g <= 1:
                            continue
                        pairs = sorted(
                            divisor_pairs(g),
                            key=lambda ab: simple_synthesis(ab[0])[0]
                            + simple_synthesis(ab[1])[0],
                        )
                        for f1, f2 in pairs:
                            factors = (
                                sorted({f1, f2}, key=lambda f: simple_synthesis(f)[0])
                                if f1 != f2
                                else (f1,)
                            )
                            for factor in factors:
                                if factor <= 1:
                                    continue

                                if (
                                    simple_synthesis(factor)[0] + cost_mul + cost_add
                                    >= best_cost
                                ):
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
                                total2 = (
                                    cost_f + cost_a2 + cost_b2 + cost_mul + cost_add
                                )
                                if total2 < best_cost - 1e-12:
                                    best_cost = total2
                                    best_expr = BinOpNode(
                                        "mul",
                                        expr_f,
                                        BinOpNode("add", expr_a2, expr_b2),
                                    )

        result = (best_cost, best_expr)
        _solve_cache[value] = (result, depth)
        return result

    if verbose:
        print("near_exact_powers:", near_exact_powers(target, max_ext, max_val))
        print(
            "candidate_add_splits:", candidate_add_splits(target, max_ext, base_powers)
        )
        cost, expr = simple_synthesis(target)
        print("simple_synthesis:", (cost, render(expr) if expr else None))
    cost, expr_tree = solve(target, depth_limit)

    if verbose:
        print(f"nodes: {nodes}, dp_hits: {dp_hits}")
    if expr_tree is not None:
        # Pipeline: tuple-based AST (expr_tree) -> string expression (render) -> simplified string (simplify_expr)
        expr = render(expr_tree)
    else:
        expr = "None"
    return cost, expr


def optimize_sum_with_U(target: list[int], U: int) -> str:
    """
    @brief API for bin_dump.pyw
    @param target Binary dump of values, which is unused directly as compression
    @param U max usable literal
    """
    total_target = sum(target)
    return simplify_expr(
        synthesize_optimal_with_exp(
            total_target,
            disallowed=set(),
            max_ext=U,
            max_val=(1 << 31) - 1,
            verbose=False,
        )[1]
    )


optimizations = [optimize_sum_with_U]

_BIN_OPS = {
    "add": int.__add__,
    "sub": int.__sub__,
    "mul": int.__mul__,
    "pow": int.__pow__,
}


def render(n: Node) -> str:
    ops = {"add": "+", "sub": "-", "mul": "*", "pow": "**"}
    match n:
        case LitNode():
            return str(n.value)
        case PowBaseNode():
            return f"({BASE}**{n.exponent})"
        case BinOpNode(op=x):
            return f"({render(n.left)}{ops[x]}{render(n.right)})"
        case _:
            raise ValueError(type(n).__name__)


def eval_node(node: Node) -> tuple[int, float]:
    """Evaluate a Node expression, return (value, computed_cost)."""
    match node:
        case LitNode():
            return node.value, COST["lit"]
        case PowBaseNode():
            return BASE**node.exponent, COST["powbase"]
        case BinOpNode(op=x):
            lv, lc = eval_node(node.left)
            rv, rc = eval_node(node.right)
            return _BIN_OPS[x](lv, rv), lc + rc + COST[x]
        case _:
            raise ValueError(f"Unknown node: {node}")


def evaluate_cost(expr: str | Node) -> tuple[int, float]:
    """Deterministic evaluator used for final verification/debug."""
    if isinstance(expr, str):
        node = parse_expr(expr)
    else:
        node = expr
    return eval_node(node)


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
_MAX_BASE_EXP = int(log(_MAX_PRECOMP_VAL, BASE))
_BASE_POWERS_ALL = []
v = BASE
for _ in range(1, _MAX_BASE_EXP + 1):
    if v > _MAX_PRECOMP_VAL:
        break
    _BASE_POWERS_ALL.append(v)
    v *= BASE

_BASE_POWERS = tuple(_BASE_POWERS_ALL)

_EXACT_POWER_LIST: list[tuple[int, int, int]] = []  # (value, base, exp)

for b in range(2, isqrt(_MAX_PRECOMP_VAL) + 1):
    p = b * b
    e = 2
    while p <= _MAX_PRECOMP_VAL:
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
                nodes[write] = BinOpNode("add", left, nodes[read])
                read += 1
            else:
                nodes[write] = left

            write += 1

        n = write

    return nodes[0]


def _simplify_tree(node: Node) -> Node:
    match node:
        case PowBaseNode():
            return LitNode(BASE**node.exponent)

        case BinOpNode(op="pow"):
            left = _simplify_tree(node.left)
            right = _simplify_tree(node.right)
            if (
                isinstance(left, LitNode)
                and left.value == BASE
                and isinstance(right, LitNode)
            ):
                return LitNode(BASE**right.value)
            if isinstance(left, LitNode) and isinstance(right, LitNode):
                try:
                    return LitNode(left.value**right.value)
                except (OverflowError, ValueError):
                    pass
            return BinOpNode("pow", left, right)

        case BinOpNode(op="add"):
            left = _simplify_tree(node.left)
            right = _simplify_tree(node.right)
            terms: list[Node] = []

            def collect(n: Node) -> None:
                if isinstance(n, BinOpNode) and n.op == "add":
                    collect(n.left)
                    collect(n.right)
                else:
                    terms.append(n)

            collect(BinOpNode("add", left, right))
            seen: set[str] = set()
            uniq: list[Node] = []
            for t in terms:
                s = render(t)
                if s not in seen:
                    seen.add(s)
                    uniq.append(t)
            if len(uniq) == 1:
                return uniq[0]
            result = uniq[0]
            for t in uniq[1:]:
                result = BinOpNode("add", result, t)
            return result

        case BinOpNode(op=("sub" | "mul") as op):
            left = _simplify_tree(node.left)
            right = _simplify_tree(node.right)
            return BinOpNode(op, left, right)

        case _:
            return node


def simplify_expr(expr: str | Node) -> str:
    """Simplify expression: fold powbase/pow to lit, flatten adds, deduplicate terms."""
    if isinstance(expr, str):
        node = parse_expr(expr)
    else:
        node = expr
    return render(_simplify_tree(node))


@cache
def divisor_pairs(n: int):
    out = []
    for a in range(2, isqrt(n) + 1):
        if n % a == 0:
            out.append((a, n // a))
    return tuple(out)


@cache
def near_exact_powers(
    value: int, max_ext: int, max_val: int
) -> tuple[tuple[int, int, int], ...]:
    r = max_ext * 2
    lo = max(0, value - r)
    hi = min(max_val, value + r)
    left = bisect_left(_EXACT_POWER_VALUES, lo)
    right = bisect_right(_EXACT_POWER_VALUES, hi)
    return tuple(_EXACT_POWER_LIST[left:right])


@cache
def candidate_add_splits(
    value: int, max_ext: int, base_powers: tuple[int]
) -> tuple[int, ...]:
    if value <= 1:
        return tuple()
    cand = set()
    max_test = max(max_ext, 32) * 2

    for x in range(1, max_test + 1):
        if x <= value:
            cand.add(x)
            cand.add(value - x)

    idx = bisect_left(base_powers, value)
    for i_ in range(-5, +2):
        i = idx + i_
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

    return tuple(sorted(cand, key=lambda x: abs(x - value / 2)))


# simplify_expr is defined above; the old ExpressionSimplifier wrapper is removed

if __name__ == "__main__":

    # ======================================================
    # Benchmark / regression configuration
    # ======================================================

    TRACKING_CONFIG = {
        "BASE": BASE,
        "COST": COST,
    }

    CONFIG_KEY = hashlib.sha256(
        json.dumps(
            TRACKING_CONFIG,
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    # ======================================================
    # Persistent regression DB
    # ======================================================

    DB_PATH = Path("known_best.json")

    try:
        with open(DB_PATH, "r") as f:
            REGRESSION_DB = json.load(f)
    except FileNotFoundError:
        REGRESSION_DB = {}

    PROFILE = REGRESSION_DB.setdefault(
        CONFIG_KEY,
        {
            "meta": TRACKING_CONFIG,
            "results": {},
        },
    )

    RESULTS = PROFILE["results"]

    # ======================================================
    # Helpers
    # ======================================================

    def random_tc():
        return (
            random.randrange(1, isqrt(2**31 - 1)),
            set(),
            26,
        )

    def update_regression_db(
        target: int,
        actual_cost: float,
        expr: str,
    ):
        key = str(target)

        simplified = simplify_expr(expr)

        prev = RESULTS.get(key)

        # ----------------------------------------------
        # New benchmark entry
        # ----------------------------------------------

        if prev is None:
            RESULTS[key] = {
                "cost": actual_cost,
                "expr": simplified,
            }

            print("  ★ New benchmark entry recorded")
            return

        prev_cost = prev["cost"]

        # ----------------------------------------------
        # Better result
        # ----------------------------------------------

        if actual_cost < prev_cost - 1e-12:
            print("\033[32m  ★ NEW BEST FOUND")

            RESULTS[key] = {
                "cost": actual_cost,
                "expr": simplified,
            }

            print(f"    old cost: {prev_cost:.12f}")
            print(f"    new cost: {actual_cost:.12f}\033[0m")

            return

        # ----------------------------------------------
        # Regression
        # ----------------------------------------------

        if actual_cost > prev_cost + 1e-12:
            print("\033[31m  ✗ REGRESSION DETECTED")

            print(f"    expected <= {prev_cost:.12f}")
            print(f"    got         {actual_cost:.12f}")

            print(f"    previous expr: {prev['expr']}\033[0m")

            return

        # ----------------------------------------------
        # Equal
        # ----------------------------------------------

        print("\033[32m  = Matches known best\033[0m")

    # ======================================================
    # Test cases
    # ======================================================

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
    ]

    # ======================================================
    # Benchmark run
    # ======================================================

    print("Testing synthesis with exponentiation patterns...")
    print("=" * 70)

    print(f"\033[33mBASE: {BASE}")
    print(f"Config profile: {CONFIG_KEY}")

    print("Tracking config:")
    pprint.pp(TRACKING_CONFIG)
    print("\033[0m")
    total_start = time.time()

    for target, disallowed, max_ext in test_cases:

        print("\n" + "=" * 70)
        print(f"Target: {target}")

        if disallowed:
            print(f"Disallowed literals: {sorted(disallowed)}")

        start = time.time()

        cost, expr = synthesize_optimal_with_exp(
            target,
            disallowed=disallowed,
            max_ext=max_ext,
            max_val=2**31 - 1,
            verbose=False,
        )

        elapsed = time.time() - start

        print(f"  Time       : {elapsed:.6f}s")
        print(f"  Cost       : {cost:.12f}")
        print(f"  Expression : {expr}")

        simplified = simplify_expr(expr)

        if simplified != expr:
            print(f"  Simplified : {simplified}")

        # ==================================================
        # Verification
        # ==================================================

        actual_value, actual_cost = evaluate_cost(expr)

        if actual_value != target:
            print("\033[31m  ✗ WRONG VALUE")
            print(f"    expected: {target}")
            print(f"    got     : {actual_value}\033[0m")
            continue

        if abs(actual_cost - cost) > 1e-12:
            print("\033[31m  ✗ WRONG COST")
            print(f"    reported: {cost:.12f}")
            print(f"    actual  : {actual_cost:.12f}\033[0m")
            continue

        print("\033[32m  ✓ Verified\033[0m")

        # ==================================================
        # Regression tracking
        # ==================================================

        update_regression_db(
            target,
            actual_cost,
            expr,
        )

    total_elapsed = time.time() - total_start

    # ======================================================
    # Save DB
    # ======================================================

    with open(DB_PATH, "w") as f:
        json.dump(
            REGRESSION_DB,
            f,
            indent=2,
            sort_keys=True,
        )

    # ======================================================
    # Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("Benchmark suite complete")

    print(f"Total time: {total_elapsed:.6f}s")

    print(f"Regression DB saved to: {DB_PATH}")
    print(f"Config profile: {CONFIG_KEY}")

    print("\nKnown best results for this profile:")

    pprint.pp(RESULTS)
