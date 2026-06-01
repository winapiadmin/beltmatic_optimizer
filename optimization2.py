"""
Beltmatic arithmetic expression synthesizer.

This module searches for low-cost arithmetic expressions for integer targets,
using add/sub/mul/pow operations weighted by COST. The search is heuristic
(BFS-on-cost for small values, depth-limited beam search for larger values).

Beltmatic (https://store.steampowered.com/app/2674590) is a game where
the player builds factories on a belt. Numbers must be synthesized from
BASE powers (2**n) using arithmetic operations with weighted costs.

No analytic closed-form solution exists for the weighted-cost optimal
expression problem, hence the DP + heuristic search approach.

COST table is the efficiency (given that powbase is already crafted up to
2^31-1) which is 1/speed except for powbase and lit if you have cached them
Also in the game the speed of division and subtraction are the same

The code assumes the operations got 100% efficiency (without considering
bottlenecks of operation speeds)
"""

from __future__ import annotations
import heapq
from typing import cast
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from math import isqrt, log
from dataclasses import dataclass
from functools import cache
from operator import add, sub, mul, pow, floordiv

BASE = 2
MAX_VALUE = (1 << 31) - 1
MIN_VALUE = -(1 << 31)
COST = {
    # Weight per operation type (lower = preferred)
    "add": 1 / 2,
    "sub": 1 / 1,
    "mul": 1 / 1,
    "div": 1 / 1,
    "pow": 1 / 1.1,
    "powbase": 0.00,
    "lit": 0.00,
}
assert COST["sub"] == COST["div"]
# DP table size limit
_DP_LIMIT = 50_000

# Exponents up to 15 (covers 2**15 = 32768, fits in 32-bit signed)
_POW_EXP_MAX = 15
# Small-value threshold for cross-multiplication optimization
_SMALL_VAL_THRESHOLD = 200

# Precompute exact BASE powers once.
_MAX_PRECOMP_VAL = MAX_VALUE
_MAX_BASE_EXP = int(log(_MAX_PRECOMP_VAL, BASE))
_BASE_POWERS_ALL: list[int] = []
v = BASE
for _ in range(1, _MAX_BASE_EXP + 1):
    if v > _MAX_PRECOMP_VAL:
        break
    _BASE_POWERS_ALL.append(v)
    v *= BASE
_BASE_POWERS: tuple[int, ...] = tuple(_BASE_POWERS_ALL)

# Precompute all exact powers (value, base, exp) up to max.
_EXACT_POWER_LIST: list[tuple[int, int, int]] = []
for b in range(2, isqrt(_MAX_PRECOMP_VAL) + 1):
    p = b * b
    e = 2
    while p <= _MAX_PRECOMP_VAL:
        _EXACT_POWER_LIST.append((p, b, e))
        e += 1
        p *= b
_EXACT_POWER_LIST.sort(key=lambda t: t[0])
_EXACT_POWER_VALUES: list[int] = [x[0] for x in _EXACT_POWER_LIST]


def reconfigure(new_base: int) -> None:
    """
    Reconfigure the global numeric base and rebuild all derived, base-dependent precomputed data.
    
    This sets the module-wide BASE to `new_base`, recomputes the maximum precomputed exponent and the sequence of precomputed base powers (up to the module constant `_MAX_PRECOMP_VAL`), and replaces the `OptimalTable` singleton so callers will rebuild any DP table under the new base.
    
    Parameters:
        new_base (int): The new integer base to use for precomputation (must be >= 2).
    """
    global BASE, _MAX_BASE_EXP, _BASE_POWERS, _optimal_table
    BASE = new_base
    _MAX_BASE_EXP = int(log(_MAX_PRECOMP_VAL, BASE))
    _BASE_POWERS_ALL.clear()
    v = BASE
    for _ in range(1, _MAX_BASE_EXP + 1):
        if v > _MAX_PRECOMP_VAL:
            break
        _BASE_POWERS_ALL.append(v)
        v *= BASE
    _BASE_POWERS = tuple(_BASE_POWERS_ALL)
    _optimal_table = OptimalTable()


_BIN_OPS: dict[str, Callable[[int, int], int]] = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": floordiv,
    "pow": pow,
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


def _tokenize(s: str) -> list[int | str]:
    """
    Convert an arithmetic expression string into a sequence of integer and operator tokens.
    
    Returns:
        tokens (list[int | str]): A list where integer literals are returned as ints and operators/parentheses are returned as strings. Recognized operator tokens: '+', '-', '*', '/', '**', '(', ')'.
    
    Raises:
        ValueError: If an unexpected character is encountered, with its position in the input.
    """
    tokens: list[int | str] = []
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
        elif c == "/":
            tokens.append("/")
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(int(s[i:j]))
            i = j
        elif c.isspace():
            i += 1
        else:
            raise ValueError(f"Unexpected character {c!r} at position {i}")
    return tokens


def _parse_primary(tokens: list[int | str], pos: int) -> tuple[Node, int]:
    """
    Parse a primary expression: a parenthesized expression, an integer literal, or a power-of-BASE form.
    
    Parameters:
        tokens (list[int | str]): Token list produced by the tokenizer.
        pos (int): Index of the next token to parse.
    
    Returns:
        tuple[Node, int]: A parsed AST node and the next token index after the primary.
    
    Raises:
        ValueError: If a parenthesized expression is missing a closing ')' or an unexpected token is encountered.
    """
    tok = tokens[pos]
    match tok:
        case "(":
            pos += 1
            node, pos = _parse_expr(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError(f"Expected ')' at position {pos}")
            pos += 1
            return node, pos
        case int():
            val = tok
            pos += 1
            if val == BASE and pos < len(tokens) and tokens[pos] == "**":
                pos += 1
                match tokens[pos]:
                    case int() as k:
                        pos += 1
                        return PowBaseNode(k), pos
                right, pos = _parse_primary(tokens, pos)
                return BinOpNode("pow", LitNode(BASE), right), pos
            return LitNode(val), pos
        case _:
            raise ValueError(f"Unexpected token: {tok}")


def _parse_power(tokens: list[int | str], pos: int) -> tuple[Node, int]:
    """
    Parse a power expression at the given token position and return the resulting AST node and the updated token position.
    
    Parameters:
    	tokens (list[int | str]): Token sequence produced by the tokenizer.
    	pos (int): Index of the first token to parse.
    
    Returns:
    	(node, new_pos): Parsed `Node` (a `LitNode`, `PowBaseNode`, or `BinOpNode("pow", ...)`) and the index of the next unconsumed token.
    
    Notes:
    	If the parsed pattern is `BASE ** k` with both operands as integer literals, a `PowBaseNode(k)` is returned instead of a generic `pow` node.
    """
    left, pos = _parse_primary(tokens, pos)
    if pos < len(tokens) and tokens[pos] == "**":
        pos += 1
        right, pos = _parse_power(tokens, pos)
        match (left, right):
            case (LitNode() as lit_node, LitNode()) if lit_node.value == BASE:
                return PowBaseNode(right.value), pos
        return BinOpNode("pow", left, right), pos
    return left, pos


def _parse_mul_expr(tokens: list[int | str], pos: int) -> tuple[Node, int]:
    """
    Parse a left-associative chain of multiplication and division expressions into an AST node.
    
    Parameters:
    	tokens (list[int | str]): Token list produced by the tokenizer.
    	pos (int): Current index in `tokens` where parsing should start.
    
    Returns:
    	tuple[Node, int]: A pair of the parsed `Node` representing the multiplication/division expression
    	and the updated token position after parsing.
    """
    left, pos = _parse_power(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("*", "/"):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_power(tokens, pos)
        left = BinOpNode("mul" if op == "*" else "div", left, right)
    return left, pos


def _parse_expr(tokens: list[int | str], pos: int) -> tuple[Node, int]:
    """
    Parse an addition/subtraction expression from a token sequence.
    
    Parameters:
    	tokens (list[int | str]): Token list produced by _tokenize; tokens may be integers, operators, or parentheses.
    	pos (int): Start index in `tokens` from which to parse.
    
    Returns:
    	(node, int): A tuple where the first element is the parsed AST `Node` representing an expression of the form `mul (('+'|'-') mul)*`, and the second element is the index of the next unconsumed token.
    """
    left, pos = _parse_mul_expr(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op = tokens[pos]
        pos += 1
        right, pos = _parse_mul_expr(tokens, pos)
        if op == "+":
            left = BinOpNode("add", left, right)
        else:
            left = BinOpNode("sub", left, right)
    return left, pos


def parse_expr(s: str) -> Node:
    """
    Parse an arithmetic expression string and return its AST as a Node.
    
    The input may contain integers, parentheses, and the operators +, -, *, /, and ** (exponentiation).
    A bare occurrence of the configured BASE followed by `**<int>` is represented as a `PowBaseNode`.
    Whitespace is ignored.
    
    Parameters:
        s (str): The expression to parse.
    
    Returns:
        Node: The root of the parsed expression tree.
    
    Raises:
        ValueError: If the input contains unexpected characters, has a syntax error, or has trailing tokens after parsing.
    """
    tokens = _tokenize(s)
    node, pos = _parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"Unexpected trailing tokens: {tokens[pos:]}")
    return node


# ======================================================
# BFS-on-cost optimal table
# ======================================================


@dataclass
class OptimalTable:
    """BFS-on-cost DP table for optimal expressions up to _DP_LIMIT."""

    cost: list[float] | None = None
    expr: list[Node | None] | None = None
    config_hash: int | None = None

    def ensure(
        self,
        max_literal: int = 26,
        max_value: int = MAX_VALUE,
    ) -> None:
        """
        Build or refresh the table of minimal-cost expressions for all nonnegative values up to _DP_LIMIT (clamped by max_value).
        
        This computes and stores three arrays on the OptimalTable instance:
        - self.cost: best-known cost for each integer value (float or inf),
        - self.expr: corresponding expression Node or None,
        - self.config_hash: hash of the configuration used to build the table.
        
        Parameters:
            max_literal (int): largest integer literal seeded directly as a LitNode.
            max_value (int): maximum allowed intermediate/target value; results above this are clamped and values beyond min(_DP_LIMIT, max_value) are not stored.
        
        Behavior notes:
            - Seeds the search with all literals 0..min(max_literal, limit) and all BASE powers ≤ limit.
            - Uses a priority queue (Dijkstra-style) to relax addition/subtraction/multiplication/power combinations and retains a nearness tie-breaker when costs are equal.
            - The method is idempotent: if called with the same configuration as previously built, it returns without rebuilding.
        """
        h = hash((BASE, tuple(sorted(COST.items())), max_literal, max_value))
        if self.cost is not None and self.config_hash == h:
            return

        max_limit = _DP_LIMIT
        limit = min(max_limit, max_value)
        INF = float("inf")
        cost_arr = [INF] * (max_limit + 1)
        expr_arr: list[Node | None] = [None] * (max_limit + 1)
        nearness_arr: list[float] = [INF] * (max_limit + 1)

        def _expr(v: int) -> Node:
            """
            Return the expression node stored at index `v`, asserting it is present.
            
            Returns:
                Node: The expression node at `expr_arr[v]`.
            
            Raises:
                AssertionError: If no expression is stored at index `v` (i.e., `expr_arr[v]` is `None`).
            """
            e = expr_arr[v]
            assert e is not None
            return e

        def _relax(nv: int, nc: float, op_name: str, left: int, right: int) -> None:
            """
            Attempt to relax the DP entry for integer value `nv` by considering a binary operation `op_name(left, right)` and update the working tables/heap if this produces a strictly better solution or an equal-cost solution with improved nearness.
            
            Parameters:
                nv (int): Candidate target integer produced by the operation (may be clamped).
                nc (float): Candidate cumulative cost for producing `nv`.
                op_name (str): Operation name used to construct the expression node (one of "add","sub","mul","div","pow").
                left (int): Integer value for the left child operand (index into DP tables).
                right (int): Integer value for the right child operand (index into DP tables).
            
            Side effects:
                - Clamps `nv` into the allowed signed range using `max_value` and discards if `nv == 0` or outside `[0, limit]`.
                - If `nc` is smaller than the current stored cost by more than 1e-12, replaces `cost_arr[nv]`, `nearness_arr[nv]`, and `expr_arr[nv]` with the new values and pushes `(nc, nv)` onto `pq`.
                - If `nc` is within 1e-12 of the stored cost and the computed nearness `abs(left-right)` is smaller than the stored nearness, updates `nearness_arr[nv]` and `expr_arr[nv]` (but does not push to `pq`).
            
            Notes:
                - Uses a floating tolerance of 1e-12 for cost comparisons.
                - Constructs the expression node as `BinOpNode(op_name, _expr(left), _expr(right))`.
            """
            if nv == 0:
                return
            if nv > max_value:
                nv = max_value
            elif nv < -max_value - 1:
                nv = -max_value - 1
            if nv > limit or nv < 0:
                return
            new_nearness = float(abs(left - right))
            old_cost = cost_arr[nv]
            if nc < old_cost - 1e-12:
                cost_arr[nv] = nc
                nearness_arr[nv] = new_nearness
                expr_arr[nv] = BinOpNode(op_name, _expr(left), _expr(right))
                heapq.heappush(pq, (nc, nv))
            elif abs(nc - old_cost) <= 1e-12 and new_nearness < nearness_arr[nv]:
                nearness_arr[nv] = new_nearness
                expr_arr[nv] = BinOpNode(op_name, _expr(left), _expr(right))

        def _init_seeds(max_literal: int, limit: int) -> set[int]:
            """
            Builds the initial set of seed integers used by the DP table.
            
            The returned set contains all integers from 0 up to min(max_literal, limit), inclusive, and all powers of the global BASE (BASE**k) that are less than or equal to limit.
            
            Parameters:
                max_literal (int): maximum literal value to include as seeds (upper bound for the contiguous range).
                limit (int): absolute upper bound for included seed values and for BASE powers.
            
            Returns:
                set[int]: seed integers for initialization.
            """
            seeds = set(range(min(max_literal, limit) + 1))
            p = BASE
            while p <= limit:
                seeds.add(p)
                p *= BASE
            return seeds

        seeds = _init_seeds(max_literal, limit)

        for s in seeds:
            cost_arr[s] = 0.0
            nearness_arr[s] = 0.0
            if s <= max_literal:
                expr_arr[s] = LitNode(s)
            else:
                # s is a BASE power; find exponent n such that BASE**n == s
                n = 0
                tmp = s
                while tmp > 1:
                    tmp //= BASE
                    n += 1
                expr_arr[s] = PowBaseNode(n)

        pq = [(0.0, s) for s in seeds]
        heapq.heapify(pq)
        seed_list = sorted(seeds)
        small_non_seeds: list[int] = []

        while pq:
            cur_cost, val = heapq.heappop(pq)
            if cur_cost > cost_arr[val]:
                continue

            for seed in seed_list:
                sc = cost_arr[seed]
                _relax(val + seed, cur_cost + sc + COST["add"], "add", val, seed)
                if val >= seed:
                    _relax(val - seed, cur_cost + sc + COST["sub"], "sub", val, seed)
                if seed >= val:
                    _relax(seed - val, cur_cost + sc + COST["sub"], "sub", seed, val)
                _relax(val * seed, cur_cost + sc + COST["mul"], "mul", val, seed)

            if val >= 2:
                for seed in seed_list:
                    if seed <= _POW_EXP_MAX:
                        bit_limit = limit.bit_length()
                        if seed * (val.bit_length() - 1) <= bit_limit - 1:
                            nv = val**seed
                            if nv <= limit:
                                _relax(
                                    nv,
                                    cur_cost + cost_arr[seed] + COST["pow"],
                                    "pow",
                                    val,
                                    seed,
                                )

            if 2 <= val <= _POW_EXP_MAX:
                for seed in seed_list:
                    if seed >= 2:
                        bit_limit = limit.bit_length()
                        if val * (seed.bit_length() - 1) <= bit_limit - 1:
                            nv = seed**val
                            if nv <= limit:
                                _relax(
                                    nv,
                                    cur_cost + cost_arr[seed] + COST["pow"],
                                    "pow",
                                    seed,
                                    val,
                                )

            if val < _SMALL_VAL_THRESHOLD and val not in seeds:
                for other in small_non_seeds:
                    if other == val:
                        continue
                    _relax(
                        val * other,
                        cur_cost + cost_arr[other] + COST["mul"],
                        "mul",
                        val,
                        other,
                    )
                    _relax(
                        val + other,
                        cur_cost + cost_arr[other] + COST["add"],
                        "add",
                        val,
                        other,
                    )

            if val not in seeds and val < _SMALL_VAL_THRESHOLD:
                small_non_seeds.append(val)

        self.cost = cost_arr
        self.expr = expr_arr
        self.config_hash = h


_optimal_table = OptimalTable()


def ensure_optimal_table(
    max_literal: int = 26,
    max_value: int = MAX_VALUE,
) -> None:
    """Ensure the singleton DP table is built for the given parameters."""
    _optimal_table.ensure(max_literal, max_value)


def optimal_lookup(value: int) -> tuple[float, Node] | None:
    """
    Lookup a precomputed minimal-cost expression for a nonnegative integer from the optimal DP table.
    
    Returns:
        A tuple `(cost, expr)` where `cost` is the minimal accumulated cost and `expr` is the corresponding `Node`, or `None` if the table has no entry for the value (e.g. table not built, value > internal DP limit, or cost is infinite).
    """
    if _optimal_table.cost is not None and value <= _DP_LIMIT:
        c = _optimal_table.cost[value]
        if c != float("inf"):
            assert _optimal_table.expr is not None
            e = _optimal_table.expr[value]
            assert e is not None
            return c, e
    return None


def synthesize_optimal_with_exp(
    target: int,
    disallowed: set[int] | None = None,
    max_ext: int = 26,
    max_val: int = MAX_VALUE,
    verbose: bool = False,
) -> tuple[float, str]:
    """
    Synthesize a low-cost arithmetic expression for an integer target.

    Minimizes weighted operation cost using add/sub/mul/pow/BASE powers.

    Parameters
    ----------
    target:
        Integer value to synthesize.
    disallowed:
        Literal constants forbidden in the expression.
    max_ext:
        Maximum literal integer value allowed to appear directly in expressions
        (values > max_ext must be synthesized from BASE powers).
    max_val:
        Maximum permitted intermediate value.
    verbose:
        Enables diagnostic logging.

    Returns
    -------
    (estimated cost, rendered expression string)

    Notes
    -----
    Heuristic and depth-limited; optimality is not globally guaranteed.
    Costs are floating-point and may accumulate rounding error.
    """

    disallowed = disallowed or set()
    nodes = 0
    dp_hits = 0

    # Keep base powers local to the current max_val.
    if max_val >= _MAX_PRECOMP_VAL:
        base_powers = _BASE_POWERS
    else:
        end = bisect_right(_BASE_POWERS_ALL, max_val)
        base_powers = tuple(_BASE_POWERS_ALL[:end])

    depth_limit = 4
    cost_add = COST["add"]
    cost_sub = COST["sub"]
    cost_mul = COST["mul"]
    cost_pow = COST["pow"]
    cost_div = COST["div"]
    cost_powbase = COST["powbase"]
    cost_lit = COST["lit"]

    ensure_optimal_table(max_ext, max_val)

    @cache
    def simple_synthesis(value: int) -> tuple[float, Node | None]:
        """
        Produce a greedy expression for an integer target using literals, exact BASE powers, an optimal DP lookup, or a binary (powers-of-two) decomposition.
        
        Returns:
            tuple[float, Node | None]: A pair `(cost, node)` where `cost` is the estimated cost of the produced expression and `node` is the AST for that expression, or `None` if no expression was synthesized (cost will be `inf` in that case).
        """
        if value <= max_ext and value not in disallowed:
            return cost_lit, LitNode(value)

        if value > 0:
            # Exact power of BASE -> PowBaseNode (cost 0)
            e = 0
            p = 1
            while p < value:
                p *= BASE
                e += 1
            if p == value:
                return cost_powbase, PowBaseNode(e)

        # DP table covers add/sub/mul for 0.._DP_LIMIT
        dp = optimal_lookup(value)
        if dp is not None:
            return dp

        # Binary decomposition via pow(2, n) — guaranteed fallback for any BASE
        if value > 0:
            bits: list[int] = []
            tmp = value
            while tmp:
                bits.append(tmp & 1)
                tmp >>= 1
            parts: list[Node] = []
            total = 0.0
            for i, bit in enumerate(bits):
                if not bit:
                    continue
                if i == 0:
                    parts.append(LitNode(1))
                    total += cost_lit
                else:
                    parts.append(BinOpNode("pow", LitNode(2), LitNode(i)))
                    total += cost_pow + cost_lit * 2
            expr = build_balanced(parts)
            if len(parts) > 1:
                total += (len(parts) - 1) * cost_add
            return total, expr

        return float("inf"), None

    _solve_cache: dict[int, tuple[tuple[float, Node | None], int]] = {}

    def solve(value: int, depth: int = depth_limit) -> tuple[float, Node | None]:
        """
        Searches for a low-cost expression that evaluates to `value` using a depth-limited heuristic search.
        
        Parameters:
            value (int): Target integer value to synthesize.
            depth (int): Remaining recursion depth for search; lower values restrict exploration.
        
        Returns:
            tuple[float, Node | None]: A pair where the first element is the estimated minimal weighted cost and the second element is the expression `Node` that achieves that cost or `None` if no expression was found.
        """
        nonlocal nodes, dp_hits
        result: tuple[float, Node | None]
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
                n = 0
                tmp = value
                while tmp > 1:
                    tmp //= BASE
                    n += 1
                result = (cost_powbase, PowBaseNode(n))
                _solve_cache[value] = (result, depth)
                return result

        if depth <= 0:
            result = simple_synthesis(value)
            _solve_cache[value] = (result, depth)
            return result

        best_cost, best_expr = simple_synthesis(value)

        _node_value_cache: dict[int, int] = {}

        def _node_value(node: Node) -> int:
            """
            Evaluate an AST node to its integer value, clamped to the current belt limits.
            
            Parameters:
                node (Node): A literal, pow-base, or binary-operation AST node.
            
            Returns:
                int: The node's integer value after clamping to at most `max_val` and at least `-max_val - 1`.
            
            Notes:
                The result is cached in the module-level `_node_value_cache` keyed by the node's id for reuse. PowBaseNode values use the current `BASE`.
            """
            match node:
                case LitNode():
                    return node.value
                case PowBaseNode():
                    return cast(int, BASE**node.exponent)
                case BinOpNode(op=x):
                    nid = id(node)
                    cached = _node_value_cache.get(nid)
                    if cached is not None:
                        return cached
                    v = _BIN_OPS[x](_node_value(node.left), _node_value(node.right))
                    # Clamp to the actual belt limit (not just MAX_VALUE)
                    if v > max_val:
                        v = max_val
                    elif v < -max_val - 1:
                        v = -max_val - 1
                    _node_value_cache[nid] = v
                    return v

        def _total_nearness(node: Node | None) -> float:
            """
            Compute the total absolute difference across all binary operation nodes in the subtree rooted at `node`.
            
            Returns:
                float: Sum of |left_value - right_value| for every `BinOpNode` contained in `node` (0.0 if `node` is None or contains no `BinOpNode`).
            """
            match node:
                case None:
                    return 0.0
                case BinOpNode():
                    my_nearness = abs(_node_value(node.left) - _node_value(node.right))
                    return (
                        my_nearness
                        + _total_nearness(node.left)
                        + _total_nearness(node.right)
                    )
                case _:
                    return 0.0

        best_nearness = _total_nearness(best_expr)

        def _update_best(
            total: float, entry: Node, top_nearness: float | None = None
        ) -> None:
            """
            Update the current best candidate by replacing nonlocal best_cost, best_expr, and best_nearness when `entry` is strictly better or a better tie-breaker.
            
            Parameters:
                total (float): total cost of `entry`.
                entry (Node): candidate expression node to consider.
                top_nearness (float | None): optional nearness contribution from an enclosing context; if provided and `entry` is a `BinOpNode`, the entry's nearness is computed as `top_nearness` plus the summed `_total_nearness` of the entry's left and right children; otherwise the entry's nearness is `_total_nearness(entry)`.
            
            Behavior:
                - Replaces the nonlocal best when `total` is smaller than the current best_cost by more than 1e-12.
                - If `total` is within 1e-12 of best_cost, replaces the best only if the computed nearness is smaller than best_nearness by more than 1e-12.
                - When replacing, updates best_cost, best_expr, and best_nearness accordingly.
            
            Side effects:
                Updates nonlocal variables `best_cost`, `best_expr`, and `best_nearness`. No value is returned.
            """
            nonlocal best_cost, best_expr, best_nearness
            if total < best_cost - 1e-12:
                entry_nearness = (
                    top_nearness
                    + _total_nearness(entry.left)
                    + _total_nearness(entry.right)
                    if top_nearness is not None and isinstance(entry, BinOpNode)
                    else _total_nearness(entry)
                )
                best_cost = total
                best_expr = entry
                best_nearness = entry_nearness
            elif abs(total - best_cost) <= 1e-12:
                entry_nearness = (
                    top_nearness
                    + _total_nearness(entry.left)
                    + _total_nearness(entry.right)
                    if top_nearness is not None and isinstance(entry, BinOpNode)
                    else _total_nearness(entry)
                )
                if entry_nearness < best_nearness - 1e-12:
                    best_cost = total
                    best_expr = entry
                    best_nearness = entry_nearness

        def _try_powbase() -> None:
            """
            Search for expressions anchored at a BASE power and update the current best expression when a lower-cost construction is found.
            
            Explores candidate expressions of the form:
            - `BASE**n` (anchor),
            - optionally ` (expr_k * BASE**n )` when a multiplier `k` > 1 is beneficial,
            - optionally followed by `+ expr_r` or `- expr_r` to adjust a residual `r = value - anchor*k`.
            
            Each candidate is cost-pruned using the available cost bounds and the `disallowed` set; when a candidate passes pruning the function obtains recursive subexpressions for `k` and/or `r`, composes the full AST node, and calls the enclosing `_update_best` to record any improvement.
            """
            nonlocal best_expr, best_nearness
            for n, anchor in enumerate(base_powers, start=1):
                if anchor == value:
                    continue
                lo = max(1, (value - max_ext * 2 + anchor - 1) // anchor)
                hi = min(
                    31,
                    (value + max_ext * 2) // anchor,
                    max_val // anchor,
                )
                for k in range(lo, hi + 1):
                    delta = value - anchor * k
                    r = abs(delta)
                    if r not in disallowed:
                        total = cost_powbase
                        expr_k: Node | None = None
                        expr_r: Node | None = None
                        if k != 1:
                            est_k = cost_powbase + cost_mul + simple_synthesis(k)[0]
                            if est_k >= best_cost:
                                continue
                            cost_k, expr_k = solve(k, depth - 1)
                            assert expr_k is not None
                            total += cost_mul + cost_k
                        if r:
                            total += cost_sub if delta < 0 else cost_add
                        if total >= best_cost:
                            continue
                        if r:
                            if total + simple_synthesis(r)[0] >= best_cost:
                                continue
                            cost_r, expr_r = solve(r, depth - 1)
                            assert expr_r is not None
                            total += cost_r
                        entry: Node
                        if k != 1:
                            entry = BinOpNode("mul", cast(Node, expr_k), PowBaseNode(n))
                        else:
                            entry = PowBaseNode(n)
                        if r:
                            entry = BinOpNode(
                                "sub" if delta < 0 else "add",
                                entry,
                                cast(Node, expr_r),
                            )
                        _update_best(total, entry)

        def _try_pow() -> None:
            """
            Attempt exponentiation-based candidates to improve the current best expression.
            
            Searches for (base ** exp) values near the target and, for each candidate, tries to build an expression of the form `base_expr ** exp_expr` optionally adjusted by adding or subtracting a residual `k` when the power differs from the target. Candidates are pruned using cost bounds and `disallowed` literals; base, exponent, and residual subexpressions are produced by recursively calling `solve`. When a cheaper expression is found, the function updates the enclosing scope's best_cost/best_expr via `_update_best`.
            
            Side effects:
            - May modify the nonlocal variables `best_cost`, `best_expr`, and `best_nearness` through `_update_best`.
            
            Notes:
            - Only considers residual `k` values with `k <= max_ext * 2` and not in `disallowed`.
            - Respects cost-based pruning using the local cost estimates (`cost_pow`, `cost_add`, `cost_sub`) and `simple_synthesis` lower bounds.
            """
            nonlocal best_cost, best_expr, best_nearness

            pow_candidates = near_exact_powers(value, max_ext, max_val)
            if len(pow_candidates) > 1:
                pow_keys = [
                    simple_synthesis(pe[1])[0] + simple_synthesis(pe[2])[0]
                    for pe in pow_candidates
                ]
                pow_candidates = tuple(
                    p for _, p in sorted(zip(pow_keys, pow_candidates))
                )
            for pow_val, base, exp in pow_candidates:
                k = abs(pow_val - value)
                if k <= max_ext * 2 and k not in disallowed:
                    if simple_synthesis(base)[0] + cost_pow >= best_cost:
                        continue
                    cost_base, expr_base = solve(base, depth - 1)
                    assert expr_base is not None
                    est_exp = cost_base + cost_pow + simple_synthesis(exp)[0]
                    if est_exp >= best_cost:
                        continue
                    cost_exp, expr_exp = solve(exp, depth - 1)
                    assert expr_exp is not None
                    op_cost = cost_sub if pow_val > value else cost_add
                    total = cost_base + cost_exp + cost_pow + op_cost * (k != 0)
                    if total >= best_cost:
                        continue
                    entry = BinOpNode("pow", expr_base, expr_exp)
                    if k:
                        if total + simple_synthesis(k)[0] >= best_cost:
                            continue
                        cost_k, expr_k = solve(k, depth - 1)
                        assert expr_k is not None
                        total += cost_k
                        entry = (
                            BinOpNode("sub", entry, expr_k)
                            if pow_val > value
                            else BinOpNode("add", entry, expr_k)
                        )
                    _update_best(total, entry)

        def _try_mul() -> None:
            """
            Attempt multiplicative factorizations of the current target value and update the best-found expression when a lower-cost multiplication is discovered.
            
            For each divisor pair (a, b) of the target (skipping factors equal to 1), the function:
            - heuristically orders candidate pairs by their simple-synthesis costs,
            - prunes pairs whose estimated partial costs cannot beat the current best cost,
            - recursively solves subproblems for `a` and `b` with one less depth,
            - if both subsolutions exist and the combined cost (cost_a + cost_b + cost for `"mul"`) improves the best known cost or ties with better nearness, constructs a `BinOpNode("mul", expr_a, expr_b)` and updates the best candidate using `_update_best`, using `abs(a - b)` as the nearness metric.
            
            Does nothing if depth or DP-limit conditions prevent further exploration.
            """
            nonlocal best_cost, best_expr, best_nearness
            if value > _DP_LIMIT and depth <= 1:
                return

            pairs = divisor_pairs(value)
            if len(pairs) > 1:
                mul_keys = [
                    simple_synthesis(a)[0] + simple_synthesis(b)[0] for a, b in pairs
                ]
                pairs = tuple(p for _, p in sorted(zip(mul_keys, pairs)))
            for a, b in pairs:
                if a == 1 or b == 1:
                    continue

                if simple_synthesis(a)[0] + cost_mul >= best_cost:
                    continue
                cost_a, expr_a = solve(a, depth - 1)
                assert expr_a is not None
                if cost_a + cost_mul >= best_cost:
                    continue
                cost_b, expr_b = solve(b, depth - 1)
                assert expr_b is not None
                total = cost_a + cost_b + cost_mul
                entry = BinOpNode("mul", expr_a, expr_b)
                _update_best(total, entry, float(abs(a - b)))

        def _try_add() -> None:
            """
            Attempt additive decompositions of the current target value and update the best-found expression.
            
            Tries candidate splits a + b = value (from candidate_add_splits), optionally orders them by their greedy/simple-synthesis cost, prunes splits that cannot improve the current best cost, recursively solves each side with reduced depth, and, when a better solution is found, updates the closed-over best_cost, best_expr, and best_nearness via _update_best. Returns immediately if the value exceeds the DP table limit and remaining search depth is 1 or less.
            """
            nonlocal best_cost, best_expr, best_nearness
            if value > _DP_LIMIT and depth <= 1:
                return

            splits = candidate_add_splits(value, max_ext, base_powers)
            if len(splits) > 1:
                add_keys = [
                    simple_synthesis(x)[0] + simple_synthesis(value - x)[0]
                    for x in splits
                ]
                splits = tuple(x for _, x in sorted(zip(add_keys, splits)))
            for a in splits:
                b = value - a
                if a == 0 or b == 0:
                    continue

                if simple_synthesis(a)[0] + cost_add >= best_cost:
                    continue
                cost_a, expr_a = solve(a, depth - 1)
                assert expr_a is not None
                if cost_a + cost_add < best_cost:
                    cost_b, expr_b = solve(b, depth - 1)
                    assert expr_b is not None
                    total = cost_a + cost_b + cost_add
                    entry = BinOpNode("add", expr_a, expr_b)
                    _update_best(total, entry, float(abs(a - b)))

        def _try_div() -> None:
            """
            Search for division-based candidate expressions of the form (BASE**n ± offset) / d and update the current best expression when a lower-cost candidate is found.
            
            Enumerates integer divisors d (2..min(max_ext,32,max_val//value)) and, for each BASE power p = BASE**n within value*d ± (d-1) and within max_ext offsets, constructs candidates:
            - exact numerator p -> (p - lo) adjustment as a subtraction when p ∈ [lo..hi],
            - numerator p < lo -> numerator = (BASE**n + offset) when offset ≤ max_ext,
            where lo = value * d and hi = lo + d - 1. Skips candidates that exceed max_val or include any literal in `disallowed`. Uses precomputed local costs (cost_powbase, cost_div, cost_add, cost_sub) to estimate candidate cost and calls _update_best(total_cost, node) to update best_cost/best_expr/best_nearness when an improvement is found.
            """
            nonlocal best_cost, best_expr, best_nearness
            if cost_div >= best_cost:
                return
            max_div = min(max_ext, 32, max_val // value) if value > 0 else 0
            for d in range(2, max_div + 1):
                if d in disallowed:
                    continue
                lo = value * d
                if lo > max_val:
                    continue
                hi = lo + d - 1
                for n, p in enumerate(base_powers, start=1):
                    if p + max_ext < lo:
                        continue
                    if p - max_ext > hi:
                        break
                    if lo <= p <= hi:
                        rem = p - lo
                        if rem and rem in disallowed:
                            continue
                        total = cost_powbase + cost_div
                        if rem:
                            total += cost_sub
                        entry: Node = PowBaseNode(n)
                        if rem:
                            entry = BinOpNode("sub", entry, LitNode(rem))
                        div_entry = BinOpNode("div", entry, LitNode(d))
                        _update_best(total, div_entry)
                    if p < lo:
                        offset = lo - p
                        if offset > max_ext or offset in disallowed:
                            continue
                        total = cost_powbase + cost_div
                        total += cost_add
                        entry = BinOpNode("add", PowBaseNode(n), LitNode(offset))
                        div_entry = BinOpNode("div", entry, LitNode(d))
                        _update_best(total, div_entry)

        if cost_powbase < best_cost:
            _try_powbase()
        if cost_pow < best_cost:
            _try_pow()
        if cost_mul < best_cost:
            _try_mul()
        if cost_add < best_cost:
            _try_add()
        if cost_div < best_cost:
            _try_div()

        result = (best_cost, best_expr)
        _solve_cache[value] = (result, depth)
        return result

    if verbose:
        print("near_exact_powers:", near_exact_powers(target, max_ext, max_val))
        print(
            "candidate_add_splits:",
            candidate_add_splits(target, max_ext, base_powers),
        )
        cost, se = simple_synthesis(target)
        print("simple_synthesis:", (cost, render(se) if se else None))
    cost, expr_tree = solve(target, depth_limit)

    if verbose:
        print(f"nodes: {nodes}, dp_hits: {dp_hits}")
    if expr_tree is not None:
        # Pipeline: tuple-based AST (expr_tree) -> string expression (render)
        expr = render(expr_tree)
    else:
        expr = "None"
    return cost, expr


def optimize_sum_with_U(target: list[int], U: int) -> str:
    """
    Synthesize an expression that represents the sum of the given integers.
    
    Parameters:
        target (list[int]): List of integers whose sum should be represented.
        U (int): Maximum literal integer allowed in the synthesized expression (passed as `max_ext`).
    
    Returns:
        expr (str): Rendered expression string that evaluates to sum(target), or the string "None" if no expression was found.
    """
    total_target = sum(target)
    return synthesize_optimal_with_exp(
        total_target,
        disallowed=set(),
        max_ext=U,
        max_val=MAX_VALUE,
        verbose=False,
    )[1]


optimizations = [optimize_sum_with_U]


def render(n: Node) -> str:
    """
    Serialize a Node AST into a parenthesized infix expression string.
    
    The output uses '+' '-' '*' '/' and '**' as infix operators and includes parentheses
    to preserve the original grouping.
    
    Returns:
        str: The rendered expression string.
    
    Raises:
        ValueError: If `n` is not a recognized Node subtype.
    """
    ops = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}
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
    """
    Compute the integer result and accumulated weighted cost of an expression AST `node`.
    
    Parameters:
        node (Node): Expression AST to evaluate (LitNode, PowBaseNode, or BinOpNode).
    
    Returns:
        tuple[int, float]: `(value, cost)` where `value` is the evaluated integer result and `cost` is the sum of operation/literal costs from `COST` as a float. Division is performed as integer floor division.
    
    Raises:
        ValueError: If `node` is not a recognized Node variant.
    """
    _local_ops: dict[str, Callable[[int, int], int]] = {
        k: op
        for k, op in [
            ("add", add),
            ("sub", sub),
            ("mul", mul),
            ("div", floordiv),
            ("pow", pow),
        ]
    }

    def _eval(n: Node) -> tuple[int, float]:
        """
        Evaluate an expression AST node and compute its integer value together with the accumulated cost.
        
        Parameters:
            n (Node): Expression node to evaluate (LitNode, PowBaseNode, or BinOpNode).
        
        Returns:
            tuple[int, float]: A pair (value, cost) where `value` is the integer result of evaluating the node and `cost` is the sum of cost contributions from this node and all child nodes.
        
        Raises:
            ValueError: If `n` is not a recognized node type.
        """
        match n:
            case LitNode():
                return n.value, COST["lit"]
            case PowBaseNode():
                return BASE**n.exponent, COST["powbase"]
            case BinOpNode(op=x):
                lv, lc = _eval(n.left)
                rv, rc = _eval(n.right)
                return _local_ops[x](lv, rv), lc + rc + COST[x]
            case _:
                raise ValueError(f"Unknown node: {n}")

    return _eval(node)


def evaluate_cost(expr: str | Node) -> tuple[int, float]:
    """
    Evaluate an expression AST or expression string and return its integer value and accumulated cost.
    
    If `expr` is a string it is parsed with `parse_expr`; if it is a `Node` it is evaluated directly.
    
    Parameters:
        expr (str | Node): An expression represented either as a source string or as a parsed `Node`.
    
    Returns:
        (int, float): Tuple `(value, cost)` where `value` is the evaluated integer result and `cost` is the total weighted cost of the expression.
    """
    if isinstance(expr, str):
        node = parse_expr(expr)
    else:
        node = expr
    return eval_node(node)


@cache
def is_power(n: int, base: int) -> bool:
    """
    Determine whether n is an exact positive integer power of base.
    
    Returns:
        True if there exists an integer k ≥ 0 such that n == base**k, False otherwise.
    """
    if n < 1 or base < 2:
        return False

    # Fast path for your actual BASE=2 case.
    if base == 2:
        return (n & (n - 1)) == 0

    while n % base == 0:
        n //= base
    return n == 1


def build_balanced(nodes: list[Node]) -> Node | None:
    """
    Constructs a balanced binary addition tree from the provided nodes.
    
    Returns:
    	A Node representing the balanced binary tree that adds all elements in `nodes`, or `None` if `nodes` is empty.
    """
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


_DIVISOR_PAIR_MAX_ITER = 2000


@cache
def divisor_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """
    List divisor pairs (a, b) for n where a >= 2 and a * b == n.
    
    Search only tests a in [2, min(isqrt(n), _DIVISOR_PAIR_MAX_ITER)] and returns found pairs as a tuple of (a, b) tuples in ascending order of a.
    
    Parameters:
        n (int): The integer to factor (assumed positive).
    
    Returns:
        tuple[tuple[int, int], ...]: Tuple of divisor pairs (a, n//a). Empty tuple if no divisors are found in the tested range.
    """
    out = []
    limit_a = isqrt(n)
    max_a = min(limit_a, _DIVISOR_PAIR_MAX_ITER)
    for a in range(2, max_a + 1):
        if n % a == 0:
            out.append((a, n // a))
    return tuple(out)


@cache
def near_exact_powers(
    value: int, max_ext: int, max_val: int
) -> tuple[tuple[int, int, int], ...]:
    """
    Find exact integer powers near a target value.
    
    Searches the precomputed exact-power table and returns triples (power_value, base, exponent)
    for powers whose value is within value ± (2 * max_ext) and not greater than max_val.
    
    Parameters:
        value (int): Target integer to search around.
        max_ext (int): Tolerance bound; powers within ±(2 * max_ext) of `value` are included.
        max_val (int): Upper bound for power values to consider.
    
    Returns:
        tuple[tuple[int, int, int], ...]: Sorted tuple of (power_value, base, exponent) triples
        whose power_value satisfies the range criteria.
    """
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
    """
    Generate candidate integers `a` (with `1 < a < value`) to consider for decomposition `a + b = value` during addition-based synthesis.
    
    Parameters:
        value (int): Target integer to split; returns an empty tuple if `value <= 1`.
        max_ext (int): External literal bound used to include small direct candidates (controls how far small integers are tested).
        base_powers (tuple[int]): Sorted tuple of BASE powers used to add nearby power- and multiple-based candidates.
    
    Returns:
        tuple[int, ...]: Sorted tuple of candidate `a` values (each in `1 < a < value`), ordered by proximity to `value / 2`.
    """
    if value <= 1:
        return tuple()
    cand = set()
    max_test = max(max_ext, 32) * 2

    for x in range(1, max_test + 1):
        if x <= value:
            cand.add(x)
            cand.add(value - x)

    idx = bisect_left(base_powers, value)
    for offset in range(-5, +2):
        i = idx + offset
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
