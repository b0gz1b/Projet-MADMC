import numpy as np
from typing import List
from dKP import *
NUMBER_OF_CHILDREN = lambda d: d + 1
MAX_LEAF_SIZE = 20

def P0(dkp: DKP, m: int) -> NDTree:
    """
    Generate m solutions mutually non-dominated
    :param dkp: the instance of the problem
    :param m: the number of solutions to generate
    :return: the NDTree containing the solutions
    """
    p0 = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE)
    for _ in range(m):
        solution = np.zeros(dkp.n,dtype=int)
        q = np.random.dirichlet(np.ones(dkp.d),size=1)
        r = dkp.R(q)
        desc_sorting = np.flip(np.argsort(r))
        solution_weight = 0
        solution_value = np.zeros(dkp.d,dtype=int)
        for i in desc_sorting:
            if solution_weight + dkp.w[i] <= dkp.W:
                solution[i] = 1
                solution_weight += dkp.w[i]
                solution_value += dkp.v[i]
        p0.update(DKPPoint(dkp, solution, weight=solution_weight, value=solution_value))
    return p0

def PLS(dkp: DKP, m: int) -> List[DKPPoint]:
    """
    Performs the pareto local search (PLS) algorithm on the instance of the problem
    :param dkp: the instance of the problem
    :param m: the size of the initial population
    :return: the approximate pareto front
    """
    initial_pop = P0(dkp, m) # Initial population
    efficient_set_approx = initial_pop.copy() # Pareto front archive
    current_pop = initial_pop.copy() # Current population
    aux_pop = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE) # Auxiliary population
    while current_pop.get_pareto_front() != []:
        for p in current_pop.get_pareto_front():
            neighbors = p.neighbors_one_one()
            for neighbor in neighbors:
                if not p.covers(neighbor):
                    if efficient_set_approx.update(neighbor):
                        aux_pop.update(neighbor)
        current_pop = aux_pop.copy()
        aux_pop = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE)

    return efficient_set_approx.get_pareto_front()
    
