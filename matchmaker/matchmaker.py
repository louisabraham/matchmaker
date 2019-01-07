__all__ = ["cost_matrix_without_inf", "linear_sum_assignment_iter"]

from heapq import heappush, heappop
import numpy as np

from .munkres import linear_sum_assignment


def cost_matrix_without_inf(cost_matrix):
    """
    Replaces inf with a value that yield the same assignment.
    inf means that an edge should not be used.
    """
    cost_matrix = np.asarray(cost_matrix).copy()
    if not np.issubdtype(cost_matrix.dtype, np.floating):
        return cost_matrix
    if np.isneginf(cost_matrix).any():
        raise ValueError("matrix contains -inf")

    values = cost_matrix[~np.isinf(cost_matrix)]
    m = values.min()
    M = values.max()
    n = min(cost_matrix.shape)
    # strictly positive constant even when added
    # to elements of the cost matrix
    positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
    place_holder = (M + (n - 1) * (M - m)) + positive
    cost_matrix[np.isposinf(cost_matrix)] = place_holder
    return cost_matrix


def _coords_to_index(x):
    if not x:
        return np.array([], dtype=int), np.array([], dtype=int)
    a, b = zip(*x)
    return np.array(a), np.array(b)


def _index_to_coords(x):
    return list(zip(*x))


def _second_best_assignment(cost_matrix, M, I, O):
    """Returns the second best solution to the assignment problem
    defined by ``cost_matrix`` with the constraints:
    
    - all edges from ``I`` must be used
    - the edges from ``O`` are forbidden

    ``M`` is the best solution to this problem.

    If the solution does not exist, returns None
    """
    cost_matrix = cost_matrix.copy()

    # select a boolean mask to discard the
    # rows and columns of already chosed assignments
    select = np.ones_like(cost_matrix, dtype=bool)
    i1, i2 = _coords_to_index(I)
    select[i1, :] = False
    select[:, i2] = False

    # put an infinite weight to forbidden edges
    cost_matrix[_coords_to_index(O)] = np.inf

    # the theorem only works for perfect matchings
    n, m = cost_matrix.shape
    assert n == m

    # build the graph
    distance = np.tile(np.inf, (2 * n, 2 * n))
    distance[:n, n:][select] = cost_matrix[select]
    distance[M[0], n + M[1]] = np.inf
    distance[n + M[1], M[0]] = np.where(
        np.isposinf(cost_matrix[M]), cost_matrix[M], -cost_matrix[M]
    )
    backtrack = np.tile(np.arange(2 * n), (2 * n, 1))

    # Floyd–Warshall algorithm
    for k in range(2 * n):
        for i in range(2 * n):
            for j in range(2 * n):
                if distance[i, k] + distance[k, j] < distance[i, j]:
                    distance[i, j] = distance[i, k] + distance[k, j]
                    backtrack[i, j] = backtrack[i, k]

    # restricting to the first n values should not be necessary
    # but not doing it can fail some tests like this one:
    # list(
    #     linear_sum_assignment_iter(
    #         [
    #             [-319.77581059, -426.02257358, 129.91680618],
    #             [201.86064154, float("inf"), 483.324417],
    #             [434.94794501, -324.91883433, 204.60857852],
    #         ]
    #     )
    # )

    # no solution is possible
    if np.isposinf(np.diagonal(distance)[:n].min()):
        return

    # compute the symetric difference of M and the cycle
    matching = set(_index_to_coords(M))
    target = np.diagonal(distance)[:n].argmin()
    next_vertex = lambda i: backtrack[i, target]
    i, j = target, next_vertex(target)
    while True:
        matching.add((i, j - n))
        i, j = j, next_vertex(j)
        matching.remove((j, i - n))
        if j == target:
            break
        i, j = j, next_vertex(j)

    return _coords_to_index(sorted(matching))


def _choose_in_difference(M, N):
    return next(
        eM for (eM, eN) in zip(_index_to_coords(M), _index_to_coords(N)) if eM != eN
    )


def linear_sum_assignment_iter(cost_matrix: np.ndarray):
    """Iterates over the solutions to the linear sum assignment problem
    in increasing order of cost

    The method used for the first solution is the Hungarian algorithm,
    also known as the Munkres or Kuhn-Munkres algorithm.
    
    The method used to find the second best solution and iterate over
    the solutions is described in [1]_, but is implemented in a slightly
    different, Dijkstra-like way. The states are represented as

    .. math: (cost(N_k), r, M_k, N_k, I_k, O_k)

    with :math:``r`` a random number used to avoid comparing the assignments.

    This function can also solve a generalization of the classic assignment
    problem where the cost matrix is rectangular. If it has more rows than
    columns, then not every row needs to be assigned to a column, and vice
    versa.
    It also supports infinite weights to represent edges that must never be
    used.

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    Yields
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.
    
    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, float("inf")], [3, 2, 2]])
    >>> from matchmaker import linear_sum_assignment_iter
    >>> it = linear_sum_assignment_iter(cost)
    >>> row_ind, col_ind = next(it)
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind].sum()
    5.0
    >>> row_ind, col_ind = next(it)
    >>> col_ind
    array([0, 1, 2])
    >>> cost[row_ind, col_ind].sum()
    6.0

    References
    ----------

    .. [1] Chegireddy, Chandra R., and Horst W. Hamacher. "Algorithms for finding
       k-best perfect matchings." Discrete applied mathematics 18, no. 2
       (1987): 155-165.

    """
    cost_matrix = np.asarray(cost_matrix)

    # make the cost_matrix square as the algorithm only works
    # for perfect matchings
    # any value other than 0 would work
    # see <https://cstheory.stackexchange.com/a/42168/43172>
    n, m = cost_matrix.shape
    if n < m:
        cost_matrix = np.concatenate(
            (cost_matrix, np.zeros((m - n, m), dtype=cost_matrix.dtype)), axis=0
        )
    elif n > m:
        cost_matrix = np.concatenate(
            (cost_matrix, np.zeros((n, n - m), dtype=cost_matrix.dtype)), axis=1
        )

    def transform(a, b):
        """transforms a solution assignment (a, b)
        back to the original matrix
        """
        mask = (a < n) & (b < m)
        return a[mask], b[mask]

    cost = lambda assignment: cost_matrix[assignment].sum()

    # linear_sum_assignment doesn't require the matrix to be square,
    # but second_best_assignment needs the best solution for a square matrix
    M1 = linear_sum_assignment(cost_matrix_without_inf(cost_matrix))
    if not np.isposinf(cost(M1)):
        yield transform(*M1)
    else:
        return

    # from now, use a copy of cost_matrix
    # with dtype float
    cost_matrix = cost_matrix.astype(float)

    I1 = []
    O1 = []
    N1 = _second_best_assignment(cost_matrix, M1, I1, O1)
    if N1 is None:
        return
    Q = [(cost(N1), np.random.rand(), M1, N1, I1, O1)]
    while Q:
        _, _, M, N, I, O = heappop(Q)
        yield transform(*N)

        e = _choose_in_difference(M, N)

        Ip, Op = I + [e], O
        Np = _second_best_assignment(cost_matrix, M, Ip, Op)
        if Np is not None:
            heappush(Q, (cost(Np), np.random.rand(), M, Np, Ip, Op))

        Ik, Ok = I, O + [e]
        Nk = _second_best_assignment(cost_matrix, N, Ik, Ok)
        if Nk is not None:
            heappush(Q, (cost(Nk), np.random.rand(), N, Nk, Ik, Ok))

