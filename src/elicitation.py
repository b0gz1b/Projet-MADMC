from dKP import DPoint
import numpy as np
import gurobipy as gp
from gurobipy import GRB

MAX_QUESTIONS = 1000

def pairwise_max_regret_ws(x: DPoint, y: DPoint, P: list[tuple[DPoint,DPoint]] = [], env: gp.Env = None) -> tuple[np.ndarray, float]:
    """
    Computes the pairwise max regret according to the weighted sum between two points.
    :param x: the first point
    :param y: the second point
    :param P: the set of known preferences
    :param env: the Gurobi environment
    :return: the pairwise max regret according to the weighted sum between two points
    """
    if env is None:
        env = gp.Env(empty = True)
        env.start()
    m = gp.Model("Pairwise max regret weighted sum", env=env)
    w = m.addMVar(shape=x.dimension, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w")
    m.setObjective(gp.quicksum(w[i] * (y - x).value[i] for i in range(x.dimension)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(w) == 1.0, name="sum_constraint")
    m.addConstrs((gp.quicksum(w[i] * (u - v).value[i] for i in range(x.dimension)) >= 0.0 for u, v in P), name="preference_constraints")

    m.update()
    m.optimize()

    if m.status == GRB.INFEASIBLE:
        return float("-inf"), None

    return w.X, m.ObjVal

def max_regret_ws(x: DPoint, Y: list[DPoint], P: np.ndarray = [], env: gp.Env = None) -> tuple[DPoint, float]:
    """
    Computes the max regret according to the weighted sum between a point and all the other alternatives.
    :param x: the point
    :param Y: the set of points
    :param P: the set of known preferences
    :return: the max regret according to the weighted sum between a point and a set of points
    """
    pmr = [pairwise_max_regret_ws(x, y, P, env)[1] for y in Y]
    return Y[np.argmax(pmr)], max(pmr)

def minimax_regret_ws(X: list[DPoint], P: np.ndarray = [], env: gp.Env = None) -> tuple[DPoint, float]:
    """
    Computes the minimax regret according to the weighted sum between a set of points.
    :param X: the set of points
    :param P: the set of known preferences
    :param env: the Gurobi environment
    :return: the minimax regret according to the weighted sum between a set of points
    """
    mr = [max_regret_ws(x, X, P, env)[1] for x in X]
    return X[np.argmin(mr)], min(mr)
    
def current_solution_strategy_ws(X: list[DPoint], dm_weights: np.ndarray, env: gp.Env = None) -> tuple[DPoint, int, list[float]]:
    """
    Computes the optimal solution according to the current solution strategy.
    :param X: the set of points
    :param dm_weights: the weights of the decision maker, unknown by the algorithm only used to simulate the decision maker
    :param env: the Gurobi environment
    :return: the optimal solution according to the current solution strategy
    """

    question_counter = 0

    P = []
    xp, mmr = minimax_regret_ws(X, P, env)
    yp = max_regret_ws(xp, X, P, env)[0]
    mmr_history = [mmr]
    while mmr > 0 and question_counter < MAX_QUESTIONS:
        question_counter += 1
        
        if xp.weighted_sum(dm_weights) > yp.weighted_sum(dm_weights):
            P.append((xp, yp))
            X.remove(yp)
        else:
            P.append((yp, xp))
            X.remove(xp)

        xp, mmr = minimax_regret_ws(X, P, env)
        yp = max_regret_ws(xp, X, P, env)[0]
        mmr_history.append(mmr)

    return xp, question_counter, mmr_history