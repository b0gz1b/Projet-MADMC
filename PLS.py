import time
import numpy as np
from typing import List
from dKP import DKP, DKPPoint
from NDTree import NDTree
from NDList import NDList

NUMBER_OF_CHILDREN = lambda d: d + 1
MAX_LEAF_SIZE = 20

def P0(dkp: DKP, m: int, verbose: bool = False) -> NDTree:
    """
    Generate m solutions mutually non-dominated
    :param dkp: the instance of the problem
    :param m: the number of solutions to generate
    :param verbose: True if the procedure should be verbose, False otherwise
    :return: the NDTree containing the solutions
    """
    p0 = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE)
    for _ in range(m):
        solution = np.zeros(dkp.n,dtype=int)
        q = np.random.dirichlet(np.ones(dkp.d))
        r = dkp.R(q)
        desc_sorting = np.flip(np.argsort(r))
        solution_weight = 0
        solution_value = np.zeros(dkp.d,dtype=int)
        for i in desc_sorting:
            if solution_weight + dkp.w[i] <= dkp.W:
                solution[i] = 1
                solution_weight += dkp.w[i]
                solution_value += dkp.v[i]
        p0.update(DKPPoint(dkp, solution, weight=solution_weight, value=solution_value), verbose=verbose)
    return p0

def PLS(dkp: DKP, m: int, verbose: int = 0, struct: str = "NDTree") -> List[DKPPoint]:
    """
    Performs the pareto local search (PLS) algorithm on the instance of the problem
    :param dkp: the instance of the problem
    :param m: the size of the initial population
    :param verbose: the verbosity level, 0 for no verbosity, 1 for local verbosity, 2 for global verbosity
    :return: the approximate pareto front
    """
    start = time.time()
    initial_pop = P0(dkp, m) # Initial population
    if struct == "NDList":
        initial_pop = NDList(dkp.d, initial_pop.get_pareto_front())
    if verbose != 0:
        print("Initial population of size", len(initial_pop.get_pareto_front()), "generated")
    efficient_set_approx = initial_pop.copy() # Pareto front archive
    current_pop = initial_pop.copy() # Current population
    if struct == "NDTree":
        aux_pop = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE) # Auxiliary population
    elif struct == "NDList":
        aux_pop = NDList(dkp.d)
    while current_pop.get_pareto_front() != []:
        start_it = time.time()
        _current_pop = current_pop.get_pareto_front()
        for i, p in enumerate(_current_pop):
            start_p = time.time()
            neighbors = p.neighbors_one_one()
            for j, neighbor in enumerate(neighbors):
                if not p.covers(neighbor):
                    if efficient_set_approx.update(neighbor, verbose = True if verbose == 2 else False):
                        aux_pop.update(neighbor, verbose = True if verbose == 2 else False)
            end_p = time.time()
            # if verbose != 0:
            #     print("Iteration {}/{}: {} neighbors generated and explored in {:.2f}s".format(i+1, len(_current_pop), len(neighbors), end_p-start_p))
        current_pop = aux_pop.copy()
        if struct == "NDTree":
            aux_pop = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE)
        elif struct == "NDList":
            aux_pop = NDList(dkp.d)
        if verbose != 0:
            print("Size of P:{}, iteration time: {:.2f} s, total time: {:.2f}".format(len(current_pop.get_pareto_front()), time.time()-start_it, time.time()-start))
    return efficient_set_approx.get_pareto_front()
    
