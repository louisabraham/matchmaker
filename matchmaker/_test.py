import numpy as np
from .matchmaker import linear_sum_assignment_iter


def gen_float_input(n, m, p=0.1):
    cost_matrix = np.random.rand(n, m) * 1000 - 500
    cost_matrix[np.random.rand(n, m) < p] = float("inf")
    if np.isinf(cost_matrix).all():
        return gen_float_input(n, m, p)
    return cost_matrix


def gen_int_input(n, m):
    cost_matrix = np.random.randint(-1000, 1000, (n, m))
    return cost_matrix


def is_sorted(costs):
    return all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1))


def test_linear_sum_assignment_iter(sizes=5, rounds=10):
    for _ in range(rounds):
        for gen in [gen_float_input, gen_int_input]:
            for n in range(1, sizes):
                for m in range(1, sizes):
                    cost_matrix = gen(n, m)
                    costs = list(
                        cost_matrix[assignment].sum()
                        for assignment in linear_sum_assignment_iter(cost_matrix)
                    )
                    assert is_sorted(costs)


if __name__ == "__main__":
    test_linear_sum_assignment_iter()
