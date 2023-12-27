import numpy as np

def generate_weights_ws(dim: int, num: int) -> list[np.ndarray]:
    """
    Generates a list of random weights, such that the sum of the weights is 1.
    :param dim: the dimension of the weights
    :param num: the number of weights
    :return: a list of random weights
    """
    return np.random.dirichlet(np.ones(dim), size=num)

if __name__ == '__main__':
    W = generate_weights_ws(5, 10)
    print(W)
    print(np.sum(W, axis=1))