from collections import deque
from typing import List, Tuple, Union

Step = Tuple[Union[int, float], Union[int, float], str, Union[int, float]]


def leastSteps(
    target: int, allowedNumbers: List[int], allowedOperators: List[str]
) -> List[Step]:
    queue = deque()
    visited = set()

    for num in allowedNumbers:
        if num == target:
            return [(0, num, "+", num)]
        queue.append((num, [(0, num, "+", num)]))
        visited.add(num)

    while queue:
        currentValue, stepsList = queue.popleft()

        for num in allowedNumbers:
            for operator in allowedOperators:
                newValue = None
                if operator == "+":
                    newValue = currentValue + num
                elif operator == "-":
                    newValue = currentValue - num
                elif operator == "*":
                    newValue = currentValue * num
                elif operator == "/":
                    if num != 0 and currentValue % num == 0:
                        newValue = currentValue // num
                if newValue is not None and newValue not in visited:
                    if newValue == target:
                        return stepsList + [(currentValue, num, operator, newValue)]
                    queue.append(
                        (
                            newValue,
                            stepsList + [(currentValue, num, operator, newValue)],
                        )
                    )
                    visited.add(newValue)

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
    powers_of_2 = [1 << i for i in range(32) if (1 << i) <= U]
    allowed_numbers = sorted(set(range(1, max(U, 1) + 1)) | set(powers_of_2))
    # Now call least_steps on the sum of the decomposition
    total_target = sum(target)

    expr = leastSteps(
        target=total_target,
        allowedNumbers=allowed_numbers,
        allowedOperators=["+", "-", "*", "/"],
    )
    return construct_exp(expr)


optimizations = [optimize_sum_with_U]
if __name__ == "__main__":

    def optimize(i: int, U: int):
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
