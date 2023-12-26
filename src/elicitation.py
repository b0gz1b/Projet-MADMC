from dKP import DPoint
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def pairwise_max_regret_ws(x: DPoint, y: DPoint, P: np.ndarray = []) -> tuple[float, np.ndarray]:
    """
    Computes the pairwise max regret according to the weighted sum between two points.
    :param x: the first point
    :param y: the second point
    :param P: the set of known preferences
    :return: the pairwise max regret according to the weighted sum between two points
    """
    env = gp.Env(empty = True)
    # env.setParam("OutputFlag", 0)
    env.start()

    m = gp.Model("Pairwise max regret weighted sum", env=env)
    w = m.addMVar(shape=x.dimension, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w")
    m.setObjective(w @ (y - x).value, GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(w) == 1, name="sum_constraint")
    m.addConstrs((w @ (u - v).value >= 0 for u, v in P), name="preference_constraints")

    m.update()
    m.optimize()

    if m.status == GRB.INFEASIBLE:
        return float("-inf"), None

    return m.ObjVal, w.X

def max_regret_ws(x: DPoint, Y: list[DPoint], P: np.ndarray = []) -> float:
    """
    Computes the max regret according to the weighted sum between a point and all the other alternatives.
    :param x: the point
    :param Y: the set of points
    :param P: the set of known preferences
    :return: the max regret according to the weighted sum between a point and a set of points
    """
    return max([pairwise_max_regret_ws(x, y, P) for y in Y])

def minimax_regret_ws(X: list[DPoint], P: np.ndarray = []) -> float:
    """
    Computes the minimax regret according to the weighted sum between a set of points.
    :param X: the set of points
    :param P: the set of known preferences
    :return: the minimax regret according to the weighted sum between a set of points
    """
    return min([max_regret_ws(x, X, P) for x in X])
    
    

