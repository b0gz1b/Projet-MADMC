import numpy as np
from itertools import combinations
from Capacity import Capacity
from typing import List

def generate_weights_ws(dim: int, num: int) -> List[np.ndarray]:
    """
    Generates a list of random weights, such that the sum of the weights is 1.
    :param dim: the dimension of the weights
    :param num: the number of weights
    :return: a list of random weights
    """
    return np.random.dirichlet(np.ones(dim), size=num)

def generate_weights_owa(dim: int, num: int) -> List[np.ndarray]:
    """
    Generates a list of random weights, such that the sum of the weights is 1 and the weights are in decreasing order.
    :param dim: the dimension of the weights
    :param num: the number of weights
    :return: a list of random weights
    """
    return np.flip(np.sort(np.random.dirichlet(np.ones(dim), size=num), axis=1), axis=1)

def generate_convex_capacity_choquet(dim: int, num: int) -> List[Capacity]:
    """
    Generates a list of random supermodular capacities.
    :param dim: the dimension of the capacities
    :param num: the number of capacities
    :return: a list of random supermodular capacities
    """
    res = []
    for _ in range(num):
        m = {"": 0}
        ms = np.random.dirichlet(np.ones(2**dim - 1))
        index = 0
        for i in range(1, dim + 1):
            for subset in combinations(range(dim), i):
                key = ",".join(map(str, subset))
                m[key] = ms[index]
                index += 1
        res.append(Capacity.from_moebius_inverse(dim, m))
    return res

def generate_capacity_choquet(dim: int, num: int) -> List[Capacity]:
    """
    Generates a list of random capacities.
    :param dim: the dimension of the capacities
    :param num: the number of capacities
    :return: a list of random capacities
    """
    res = []
    for _ in range(num):
        d = {"": 0, ",".join(map(str, range(dim))): 1}
        ds = np.random.uniform(size=2**dim - 1)
        index = 0
        for i in range(1, dim):
            for subset in combinations(range(dim), i):
                key = ",".join(map(str, subset))
                d[key] = ds[index]
                index += 1
        res.append(Capacity(dim, d))
    return res

def simulate_decision_makers(dim: int, num: int, pref_model: str = "ws"):
    """
    Simulates a list of decision makers.
    :param dim: the dimension of the capacities
    :param num: the number of capacities
    :param pref_model: the preference model, either "ws", "owa" or "choquet"
    :return: a list of decision makers
    """
    if pref_model == "ws":
        return generate_weights_ws(dim, num)
    elif pref_model == "owa":
        return generate_weights_owa(dim, num)
    elif pref_model == "choquet":
        return generate_convex_capacity_choquet(dim, num)
    else:
        raise Exception("Unknown preference model: {}".format(pref_model))

if __name__ == '__main__':
    d = 3
    n = 1
    c = generate_capacity_choquet(d, n)
    for n_i in range(n):
        cap = c[n_i]
        for i in range(d+1):
            for subset in combinations(range(d), i):
                print("("+",".join(map(str, subset))+")", cap.v(subset))
        print("Moebius inverse:")
        t = 0
        for i in range(d+1):
            for subset in combinations(range(d), i):
                print("("+",".join(map(str, subset))+")", cap.m(subset))
                t += cap.m(subset)
        print("Sum:", t)
        # test reverse Moebius inverse
        cap2 = Capacity.from_moebius_inverse(d, cap.moebius_inverse)
        print("inverse Moebius inverse:")
        for i in range(d+1):
            for subset in combinations(range(d), i):
                print("("+",".join(map(str, subset))+")", cap2.v(subset))
            