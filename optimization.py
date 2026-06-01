from typing import List, Tuple, Union

Step = Tuple[Union[int, float], Union[int, float], str, Union[int, float]]


def leastSteps(
    target: int, allowedNumbers: List[int], allowedOperators: List[str]
) -> List[Step]:
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
