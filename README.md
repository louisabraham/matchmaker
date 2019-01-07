[![Build
Status](https://travis-ci.org/louisabraham/matchmaker.svg?branch=master)](https://travis-ci.org/louisabraham/matchmaker)

# matchmaker

Iterate over the solutions to a linear sum assignment problem in
increasing order of cost.

# Usage

``` pycon
>>> import numpy as np
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
```

# Installation

    pip install matchmaker

# Testing

    pytest

# References

  - Chegireddy, Chandra R., and Horst W. Hamacher. "Algorithms for
    finding k-best perfect matchings." Discrete applied mathematics 18,
    no. 2 (1987): 155-165.
    [\[pdf\]](https://core.ac.uk/download/pdf/82129717.pdf)
