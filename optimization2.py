import math
import ast
from typing import Optional, Tuple, Dict, Set, List
from collections import defaultdict

BASE=10
COST = {
    "add": 1/2,
    "sub": 1/1,
    "mul": 1/1,
    "exp": 1 / 1.1,
    "powbase": 0.0,
    "lit": 0.0,
}

def evaluate_cost(expr: str) -> tuple[int, float]:
    """Deterministic AST evaluator used for final verification/debug."""
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
                # special-case BASE**n as powbase
                if isinstance(node.left, ast.Constant) and node.left.value == BASE and isinstance(node.right, ast.Constant):
                    return BASE ** node.right.value, COST["powbase"]
                return lv ** rv, lc + rc + COST["exp"]
            raise ValueError("Unsupported op")
        elif isinstance(node, ast.Constant):
            v = node.value

            return v, COST["lit"]
        else:
            raise ValueError("Unsupported AST node")

    t = ast.parse(expr, mode="eval")
    return _eval(t.body)
def is_power(n, base):
    if n < 1 or base < 2:
        return False
    while n % base == 0:
        n //= base
    return n == 1
def synthesize_optimal_with_exp(
    target: int,
    disallowed: Optional[Set[int]] = None,
    max_ext: int = 26,
    max_val: int = (1 << 31) - 1,
    verbose: bool = False
) -> Tuple[float, str]:
    """
    Optimal synthesis with exponentiation patterns and hierarchical decomposition.
    Uses memoization with limited scope to avoid memory explosion.
    """
    if disallowed is None:
        disallowed = set()
    
    # Cache for memoization (limited to prevent memory explosion)
    cache = {}
    
    def solve(value: int, depth: int = 0, max_depth: int = 2) -> Tuple[float, str]:
        """Recursive solver with depth limit."""
        # Return cached result if available
        if (value,depth) in cache:
            return cache[(value,depth)]
        
        if depth >= max_depth:
            # Base case: use simple synthesis
            return simple_synthesis(value)
        
        # Helper to check if a value can be used as literal
        def is_allowed_literal(v: int) -> bool:
            if v == 0 or v == 1:
                return True
            if v in disallowed:
                return False
            if is_power(v, BASE):  # Power of BASE
                return True
            return v <= max_ext
        
        # If value is an allowed literal, use it
        if is_allowed_literal(value) and not is_power(value,BASE):
            return COST["lit"], str(value)
        
        # Start with simple synthesis as baseline
        best_cost, best_expr = simple_synthesis(value)
        
        # Try breaking value into a * b
        # Limit search to avoid explosion
        limit = int(math.sqrt(value))

        for a in range(2, limit + 1):
            if a in disallowed:
                continue
            if value % a != 0:
                continue

            b = value // a
            if b in disallowed:
                continue

            # solve smaller first (better cache + depth behavior)
            if a <= b:
                cost_a, expr_a = solve(a, depth + 1, max_depth)
                cost_b, expr_b = solve(b, depth + 1, max_depth)
            else:
                cost_b, expr_b = solve(a, depth + 1, max_depth)
                cost_a, expr_a = solve(b, depth + 1, max_depth)

            total_cost = cost_a + cost_b + COST["mul"]

            if total_cost < best_cost:
                best_cost = total_cost
                best_expr = f"({expr_a}*{expr_b})"
        
        # Try breaking value into base ** exp
        for base in range(2, min(100, value)):
            if base in disallowed:
                continue
            
            # Find integer exponent
            exp = 2
            while True:
                power = base ** exp
                if power > value:
                    break
                if power == value:
                    cost_base, expr_base = solve(base, depth + 1, max_depth)
                    cost_exp, expr_exp = solve(exp, depth + 1, max_depth)
                    
                    total_cost = cost_base + cost_exp + COST["exp"]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_expr = f"({expr_base}**{expr_exp})"
                    break
                exp += 1
                if exp > int(math.log(max_val,BASE)):  # Limit exponent size
                    break
        
        # Try breaking value into a + b
        # Only try promising splits (around powers of two, etc.)
        for k in range(1,100):
            a = value//k
            b = value - a
            if a <= 0 or b <= 0:
                continue
            
            cost_a, expr_a = solve(a, depth + 1, max_depth)
            cost_b, expr_b = solve(b, depth + 1, max_depth)
            
            total_cost = cost_a + cost_b + COST["add"]
            if total_cost < best_cost:
                best_cost = total_cost
                best_expr = f"({expr_a}+{expr_b})"
            for f in range(2, min(20, min(a, b) + 1)):
                if a % f == 0 and b % f == 0:
                    a2 = a // f
                    b2 = b // f

                    cost_f, expr_f = solve(f, depth + 1, max_depth)
                    cost_a2, expr_a2 = solve(a2, depth + 1, max_depth)
                    cost_b2, expr_b2 = solve(b2, depth + 1, max_depth)

                    total_cost = cost_f + cost_a2 + cost_b2 + COST["mul"] + COST["add"]

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_expr = f"({expr_f}*({expr_a2}+{expr_b2}))"
        # Try breaking value into BASE**n ± k
        for n in range(2, int(math.log(max_val,BASE))):
            base = BASE**n
            if base > max_val:
                continue
            
            if base > value:
                k = base - value
                if k <= max_ext and k not in disallowed:
                    cost_k, expr_k = solve(k, depth + 1, max_depth)
                    total_cost = COST["powbase"] + cost_k + COST["sub"]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_expr = f"(({BASE}**{n})-{expr_k})"
            else:
                k = value - base
                if k <= max_ext and k not in disallowed:
                    cost_k, expr_k = solve(k, depth + 1, max_depth)
                    total_cost = COST["powbase"] + cost_k + COST["add"]
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_expr = f"(({BASE}**{n})+{expr_k})"
        
        # Try breaking value into (a**b) ± k
        for base in range(2, min(15, value)):
            if base in disallowed:
                continue
            
            for exp in range(2, min(8, int(math.log(value * 2) / math.log(base)) + 1)):
                power = base ** exp
                if power > max_val:
                    break
                
                if power > value:
                    k = power - value
                    if k <= max_ext and k not in disallowed:
                        cost_base, expr_base = solve(base, depth + 1, max_depth)
                        cost_exp, expr_exp = solve(exp, depth + 1, max_depth)
                        cost_k, expr_k = solve(k, depth + 1, max_depth)
                        
                        total_cost = cost_base + cost_exp + cost_k + COST["exp"] + COST["sub"]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_expr = f"(({expr_base}**{expr_exp})-{expr_k})"
                else:
                    k = value - power
                    if k <= max_ext and k not in disallowed:
                        cost_base, expr_base = solve(base, depth + 1, max_depth)
                        cost_exp, expr_exp = solve(exp, depth + 1, max_depth)
                        cost_k, expr_k = solve(k, depth + 1, max_depth)
                        
                        total_cost = cost_base + cost_exp + cost_k + COST["exp"] + COST["add"]
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_expr = f"(({expr_base}**{expr_exp})+{expr_k})"
        # Cache and return result
        cache[(value,depth)] = (best_cost, best_expr)
        return best_cost, best_expr
    def simple_synthesis(value: int) -> Tuple[float, str]:
        def compress_streak(start: int, end: int, BASE: int, COST) -> Tuple[float, str]:
            """
            Compress b^start + ... + b^end into best form.
            Returns (cost, expr)
            """
            v = end - start

            # --- Candidate 1: raw sum ---
            parts = []
            for p in range(start, end + 1):
                parts.append(f"({BASE}**{p})")

            def build_balanced(lst):
                if len(lst) == 1:
                    return lst[0]
                mid = len(lst) // 2
                return f"({build_balanced(lst[:mid])}+{build_balanced(lst[mid:])})"

            raw_expr = build_balanced(parts)
            raw_cost = len(parts) * COST["powbase"] + (len(parts) - 1) * COST["add"]

            best_expr = raw_expr
            best_cost = raw_cost

            # --- Candidate 2: geometric form ---
            if v >= 2:
                # numerator: (BASE^(v+1) - 1)
                num_expr = f"(({BASE}**{v+1})-1)"
                num_cost = COST["powbase"] + COST["sub"]

                # multiply by BASE^start
                if start == 0:
                    geom_expr = num_expr
                    geom_cost = num_cost
                else:
                    geom_expr = f"(({BASE}**{start})*{num_expr})"
                    geom_cost = COST["powbase"] + num_cost + COST["mul"]

                # handle division (if supported)
                if BASE - 1 != 1:
                    if "div" in COST:
                        geom_expr = f"({geom_expr}//{BASE-1})"
                        geom_cost += COST["div"]
                    else:
                        # no division → abort this candidate
                        geom_expr = None

                if geom_expr is not None and geom_cost < best_cost:
                    best_expr = geom_expr
                    best_cost = geom_cost

            return best_cost, best_expr
        if value == 0:
            return COST["lit"], "0"

        # --- power check ---
        if value > 0:
            n = int(round(math.log(value, BASE)))
            if BASE >= 2 and BASE ** n == value:
                return COST["powbase"], f"({BASE}**{n})"

        # --- BASE-N decomposition ---
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
            return COST["lit"], "0"

        # sort by power
        digits.sort(key=lambda x: x[1])

        parts = []
        total_cost = 0.0

        i = 0
        while i < len(digits):
            digit, start = digits[i]

            # only streak-compress if digit == 1
            if digit == 1:
                j = i
                while (
                    j + 1 < len(digits)
                    and digits[j + 1][0] == 1
                    and digits[j + 1][1] == digits[j][1] + 1
                ):
                    j += 1

                end = digits[j][1]

                # --- use your streak compressor ---
                cost_s, expr_s = compress_streak(start, end, BASE, COST)
                parts.append(expr_s)
                total_cost += cost_s

                i = j + 1
            else:
                # normal term: digit * BASE^power
                if start == 0:
                    expr = str(digit)
                    cost = COST["lit"]
                else:
                    expr = f"({digit}*({BASE}**{start}))"
                    cost = COST["powbase"] + COST["mul"]

                parts.append(expr)
                total_cost += cost
                i += 1

        # --- combine ---
        def build_balanced(parts_list):
            if len(parts_list) == 1:
                return parts_list[0]
            mid = len(parts_list) // 2
            return f"({build_balanced(parts_list[:mid])}+{build_balanced(parts_list[mid:])})"

        expr = build_balanced(parts)

        if len(parts) > 1:
            total_cost += (len(parts) - 1) * COST["add"]

        # --- small product shortcut ---
        if value <= max_ext * max_ext:
            for a in range(2, min(max_ext, int(math.sqrt(value)) + 1)):
                if a in disallowed:
                    continue
                if value % a == 0:
                    b = value // a
                    if b <= max_ext and b not in disallowed:
                        product_cost = COST["mul"]
                        if product_cost < total_cost:
                            return product_cost, f"({a}*{b})"

        return total_cost, expr
    
    # Start with recursive solving
    cost, expr = solve(target)
    
    # Verify
    #actual_value, actual_cost = evaluate_cost(expr)
    #if actual_value != target:
    #    # Fallback to simple synthesis
    #    cost, expr = simple_synthesis(target)
    #    actual_value, actual_cost = evaluate_cost(expr)
    
    # Make sure cost is accurate
    #if abs(cost - actual_cost) > 0.001:
    #    cost = actual_cost
    
    return cost, expr
def optimize_sum_with_U(target: list[int], U: int) -> str:
    powers_of_2 = [1 << i for i in range(32) if (1 << i) <= U]
    allowed_numbers = sorted(set(range(1, max(U, 1) + 1)) | set(powers_of_2))
    # Now call least_steps on the sum of the decomposition
    total_target = sum(target)
    cost, expr=synthesize_optimal_with_exp(
        total_target,
        disallowed={10},
        max_ext=U,
        max_val=(1 << 31) - 1,
        verbose=False,
    )
    return expr
optimizations = [optimize_sum_with_U]
# Test the enhanced algorithm
if __name__ == "__main__":
    import time
    
    test_cases = [
        (100, set(), 26),
        (22899, {10}, 26),
        (1000, set(), 26),
        (123, set(), 26),
        (319216, set(), 26),
        (1073741824, set(), 26),  # 2^30
        (1073741825, set(), 26),  # 2^30 + 1
        (2147483647, set(), 26),  # 2^31 - 1
        (1048575, set(), 26),     # 2^20 - 1
        (654321, set(), 26),      # Has contiguous sequences
        (999999, set(), 26),      # Another test
        (4953, {10}, 26),
        (4983, {10}, 26),
        (36, set(), 26),          # 6**2
        (64, set(), 26),          # 4**3 or 8**2
        (125, set(), 26),         # 5**3
        (216, set(), 26),         # 6**3
        (6896,set(),26)
    ]
    
    print("Testing synthesis with exponentiation patterns...")
    print("=" * 60)
    print(f"BASE: {BASE}")
    print(f"Cost configuration: {COST}")
    print(f"exp cost: {COST['exp']:.3f}")
    
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
                print(f"  Alternative: ((2**4) * (((2**6)+(2**3))-1) * ((2**8)+(5**2))) = cost {evaluate_cost('((2**4) * (((2**6)+(2**3))-1) * ((2**8)+(5**2)))')[1]:.3f}")
            elif target == 6896:
                print(f"  Alternative: (2**4 * (2**9 - 3**4)) = cost {evaluate_cost('2**4 * (2**9 - 3**4)')[1]:.3f}")                
        else:
            print(f"  ✗ Wrong value: {actual_value} != {target}")
