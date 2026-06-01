from typing import List, Tuple, Union

Step = Tuple[Union[int, float], Union[int, float], str, Union[int, float]]


def leastSteps(
    target: int, allowedNumbers: List[int], allowedOperators: List[str]
) -> List[Step]:
    """
    Finds a sequence of arithmetic operations using the provided numbers and operators that evaluates to `target`.
    
    Parameters:
        target (int): The integer value to reach.
        allowedNumbers (List[int]): Numbers that may be used as operands at each step.
        allowedOperators (List[str]): Operators allowed between values; expected members are "+", "-", "*", and "/".
    
    Returns:
        List[Step]: A list of steps describing the expression that evaluates to `target`, where each step is a tuple (start, num, op, result). Steps are ordered from the initial value to the final result. Division is allowed only when the divisor is nonzero and divides the dividend evenly (uses integer division). Returns an empty list if no sequence of allowed operations produces `target`.
    """
    step_data = []  # (value, prev_cv, prev_num, prev_op, parent_idx)
    visited = set()

    for num in allowedNumbers:
        if num == target:
            return [(0, num, "+", num)]
        step_data.append((num, 0, num, "+", -1))
        visited.add(num)

    head = 0
    while head < len(step_data):
        cv = step_data[head][0]
        for num in allowedNumbers:
            for operator in allowedOperators:
                newValue = None
                if operator == "+":
                    newValue = cv + num
                elif operator == "-":
                    newValue = cv - num
                elif operator == "*":
                    newValue = cv * num
                elif operator == "/":
                    if num != 0 and cv % num == 0:
                        newValue = cv // num
                if newValue is not None and newValue not in visited:
                    if newValue == target:
                        steps: List[Step] = [(cv, num, operator, newValue)]
                        idx = head
                        while idx >= 0:
                            val, prev_cv, prev_num, prev_op, parent_idx = step_data[idx]
                            steps.append((prev_cv, prev_num, prev_op, val))
                            idx = parent_idx
                        steps.reverse()
                        return steps
                    step_data.append((newValue, cv, num, operator, head))
                    visited.add(newValue)
        head += 1

    return []


def construct_exp(expr: List[Step]) -> str:
    """
    Build a parenthesized expression string from steps.
    Steps are [(start, n1, op1, r1), (r1, n2, op2, r2), ...]
    Result: '((start op1 n1) op2 n2)...'
    """
    if not expr:
        return ""

    # start from the initial value (first step's start)
    _expr = str(expr[0][0])

    for start, n, op, result in expr:
        _expr = f"({_expr}{op}{n})"

    return _expr


def optimize_sum_with_U(target: list[int], U: int) -> str:
    """
    Builds a fully parenthesized arithmetic expression that evaluates to the sum of the integers in `target`.
    
    The function computes total_target = sum(target), then constructs an allowed number set consisting of all integers from 1 to max(U, 1) together with powers of two up to limit = max(abs(total_target), U) * 2. It then searches for a sequence of operations using those numbers and the operators "+", "-", "*", "/" to reach total_target and returns the resulting parenthesized expression string produced by construct_exp. If no sequence is found, an empty string is returned.
    
    Parameters:
        target (list[int]): List of integers whose sum is the expression target.
        U (int): Upper bound used to include small integers 1..U in the allowed numbers set.
    
    Returns:
        str: A fully parenthesized arithmetic expression that evaluates to sum(target), or an empty string if no expression is found.
    """
    total_target = sum(target)
    limit = max(abs(total_target), U) * 2
    powers_of_2 = []
    v = 1
    while v <= limit:
        powers_of_2.append(v)
        v <<= 1
    allowed_numbers = sorted(set(range(1, max(U, 1) + 1)) | set(powers_of_2))

    expr = leastSteps(
        target=total_target,
        allowedNumbers=allowed_numbers,
        allowedOperators=["+", "-", "*", "/"],
    )
    return construct_exp(expr)


optimizations = [optimize_sum_with_U]
if __name__ == "__main__":

    def optimize(i: int, U: int) -> str:
        """
        Builds an optimized arithmetic expression that evaluates to the integer `i`, using `U` as the optimizer parameter.
        
        Parameters:
            i (int): The integer to represent; its binary decomposition determines the multiset of powers of two used.
            U (int): Upper-bound parameter passed to the optimizer that influences allowed numbers.
        
        Returns:
            expr (str): A fully parenthesized arithmetic expression string that evaluates to `i`.
        """
        powers = []
        p = 0
        temp = i
        while temp > 0:
            if temp & 1:
                powers.append(2**p)
            temp >>= 1
            p += 1
        return optimizations[0](powers, U)

    print(optimize(999999, 26))
