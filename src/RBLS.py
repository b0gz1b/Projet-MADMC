import time
import numpy as np
import gurobipy as gp
import random
from NDTree import NDTree
from PLS import PLS
from DKP import DKP
from DKPPoint import DKPPoint
from utils import generate_weights_ws
from PL_optimal import opt_ws
from typing import List
from elicitation import current_solution_strategy, minimax_regret, max_regret


MAX_QUESTIONS = 100
MAX_ITERATIONS = 20
NUMBER_OF_CHILDREN = lambda d: d + 1
MAX_LEAF_SIZE = 20


def computeInitSolution(dkp : DKP) -> DKPPoint:
    """
    Compute an initial solution for DKP (Greedy algorithm)
    :param dkp: the instance of the problem
    :return: the NDTree containing the DKPPoint solution
    """
    solution = np.zeros(dkp.n,dtype=int)
    solution_weight = 0
    arithmetic_mean = dkp.arithmetic_mean()
    ind_ae_max = np.argmax(arithmetic_mean)
    w_ae_max = dkp.w[ind_ae_max]
    
    while (solution_weight + w_ae_max <= dkp.W):
        solution[ind_ae_max] = 1
        solution_weight += w_ae_max
        arithmetic_mean[ind_ae_max] = 0
        ind_ae_max = np.argmax(arithmetic_mean)
        w_ae_max = dkp.w[ind_ae_max]

    return DKPPoint(dkp, solution, weight=solution_weight)
    
def computeAllSwap(dkp : DKP, x : DKPPoint, verbose: bool = False) -> List[DKPPoint]:
    """
    Compute all the neighnors (1-1) and any solution that is Pareto-dominated by another solution is removed 
    :param dkp: the instance of the problem
    :param x: the point
    :param verbose: True if the procedure should be verbose, False otherwise
    :return: the pareto front
    """
    p0 = NDTree(dkp.d, NUMBER_OF_CHILDREN(dkp.d), MAX_LEAF_SIZE)
    p0.update(x, verbose=verbose)
    neighbors = x.neighbors_one_one()
    for neighbor in neighbors:
        p0.update(DKPPoint(dkp, neighbor.x), verbose=verbose)
    return p0.get_pareto_front()

def RBLS(dkp : DKP, dm, pref_model :  str = "owa", env: gp.Env = None)-> NDTree:

    if pref_model == "ws":
        ev = lambda x: x.weighted_sum(dm)
    elif pref_model == "owa":
        ev = lambda x: x.owa(dm)
    elif pref_model == "choquet":
        ev = lambda x: x.choquet(dm)
    
    x = computeInitSolution(dkp)
    
    it = 0
    improve = True
    P = []
    
    question_counter = 0
    while improve and (it < MAX_ITERATIONS):
        X = computeAllSwap(dkp, x)
                         
        xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
        yp, _ = max_regret(xp, X, P, pref_model=pref_model, env=env)
        # _, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
        mmr_history = [mmr]
        while mmr > 0  and question_counter < MAX_QUESTIONS:
            question_counter += 1
            print("It {}\tQuestion {}: {} vs {}".format(it, question_counter, xp, yp))
            if ev(xp) > ev(yp):
                P.append((xp, yp))
                X.remove(yp)
            else:
                P.append((yp, xp))
                X.remove(xp)
            xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
            yp, _ = max_regret(xp, X, P, pref_model=pref_model, env=env)
            mmr_history.append(mmr)
            
        if x == xp:
            improve = False
        else:
            x = xp
            it += 1    
    return x, question_counter, mmr_history

if __name__ == '__main__':
    dkp = DKP.from_file("data/2KP200-TA-0.dat")
    sdkp = dkp.subinstance(30, 4, shuffle=True)
    dm = generate_weights_ws(sdkp.d, 1)[0]
    x, question_counter, mmr_history = RBLS(sdkp, dm, pref_model="ws")
    print(question_counter)
    # calculate error to optimal solution in %
    opt, _ = opt_ws(sdkp, dm)
    print(("Error: {:.2f}%".format(((opt - x.weighted_sum(dm)) / opt) * 100)))
    # compare the number of questions to the current solution strategy
    xp, question_counter, mmr_history = current_solution_strategy(PLS(sdkp, 10), dm, pref_model="ws")
    print(question_counter)
    # calculate error to optimal solution in %
    print(("Error: {:.2f}%".format(((opt - xp.weighted_sum(dm)) / opt) * 100)))
