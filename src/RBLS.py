import time
import numpy as np
import gurobipy as gp
import random
from NDTree import NDTree
from DKP import DKP
from DKPPoint import DKPPoint
from NDList import NDList
from typing import List
from elicitation import minimax_regret, max_regret


MAX_QUESTIONS = 100
MAX_ITERATIONS = 20
NUMBER_OF_CHILDREN = lambda d: d + 1
MAX_LEAF_SIZE = 20


def ComputeInitSolution(dkp : DKP) -> DKPPoint:
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
    
def ComputeAllSwap(dkp : DKP, x : DKPPoint, verbose: bool = False) -> List[DKPPoint]:
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
    
    x = ComputeInitSolution(dkp)
    
    it = 0
    improve = True
    P = []
    
    question_counter = 0
    while improve and (it < MAX_ITERATIONS):
        X = ComputeAllSwap(dkp, x)
                         
        xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
        yp, _ = max_regret(xp, X, P, pref_model=pref_model, env=env)
        # _, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
        mmr_history = [mmr]
        print("\t\tMMR: {}".format(mmr))
        # print("it: "+str(it))
        while mmr > 0  and question_counter < MAX_QUESTIONS:
            question_counter += 1            
            print("\t\tQuestion {}: {} vs {}".format(question_counter, xp, yp))
            if ev(xp) > ev(yp):
                P.append((xp, yp))
                X.remove(yp)
            else:
                P.append((yp, xp))
                X.remove(xp)
            # print("lenX: "+str(len(X)))
            xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
            yp, _ = max_regret(xp, X, P, pref_model=pref_model, env=env)
            mmr_history.append(mmr)
            print("\t\tMMR: {}".format(mmr))
            
        _, mr = max_regret(x, X, P, pref_model=pref_model, env=env)
        # print("mr : "+str(mr))
        if mr > 0:
            xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
            x = xp
            it += 1
        else:
            improve = False
    return x, question_counter, mmr_history

