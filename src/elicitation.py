from Capacity import Capacity
from DPoint import DPoint
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations

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
    start = time.time()
    delta = (y - x).value
    if env is None:
        env = gp.Env(empty = True)
        env.setParam('OutputFlag', 0)
        env.start()
    m = gp.Model("Pairwise max regret weighted sum", env=env)
    w = m.addVars(x.dimension, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w")
    m.update()
    m.setObjective(gp.quicksum(w[i] * delta[i] for i in range(x.dimension)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(w) == 1.0, name="sum_constraint")
    m.addConstrs((gp.quicksum(w[i] * (u - v).value[i] for i in range(x.dimension)) >= 0.0 for u, v in P), name="preference_constraints")

    m.optimize()

    if m.status == GRB.INFEASIBLE:
        return float("-inf"), None
    # print("Pairwise max regret weighted sum: {} in {}s".format(m.ObjVal, time.time()-start))
    return [w[i].x for i in range(x.dimension)], m.ObjVal

def pairwise_max_regret_owa(x: DPoint, y: DPoint, P: list[tuple[DPoint,DPoint]] = [], env: gp.Env = None) -> tuple[np.ndarray, float]:
    """
    Computes the pairwise max regret according to the ordered weighted average between two points.
    :param x: the first point
    :param y: the second point
    :param P: the set of known preferences
    :param env: the Gurobi environment
    :return: the pairwise max regret according to the ordered weighted average between two points
    """
    start = time.time()
    delta = np.sort(y.value) - np.sort(x.value)
    if env is None:
        env = gp.Env(empty = True)
        env.setParam('OutputFlag', 0)
        env.start()
    m = gp.Model("Pairwise max owa", env=env)
    w = m.addVars(x.dimension, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w")
    m.update()
    m.setObjective(gp.quicksum(w[i] * delta[i] for i in range(x.dimension)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(w) == 1.0, name="sum_constraint")
    for u, v in P:
        delta_P = np.sort(u.value) - np.sort(v.value)
        m.addConstr(gp.quicksum(w[i] * delta_P[i] for i in range(x.dimension)) >= 0.0, name="preference_constraints")
    for i in range(x.dimension - 1):
        m.addConstr(w[i] - w[i+1] >= 0, name="order_constraints")
    m.optimize()

    if m.status == GRB.INFEASIBLE:
        return float("-inf"), None
    # print("Pairwise max regret weighted sum: {} in {}s".format(m.ObjVal, time.time()-start))
    return [w[i].x for i in range(x.dimension)], m.ObjVal

def pairwise_max_regret_choquet(x: DPoint, y: DPoint, P: list[tuple[DPoint,DPoint]] = [], env: gp.Env = None) -> tuple[np.ndarray, float]:
    """
    Computes the pairwise max regret according to the Choquet integral between two points.
    :param x: the first point
    :param y: the second point
    :param P: the set of known preferences
    :param env: the Gurobi environment
    :return: the pairwise max regret according to the Choquet integral between two points
    """
    start = time.time()
    x_bar = [0]
    y_bar = [0]
    c_bar = [0]
    Pc_bar = [[0] for _ in range(len(P))]
    subsets = [set()]
    for size_B in range(1, x.dimension + 1):
        for i, B in enumerate(combinations(range(x.dimension), size_B)):
            subsets.append(set(B))
            x_bar.append(min(x.value[list(B)]))
            y_bar.append(min(y.value[list(B)]))
            c_bar.append(y_bar[i+1] - x_bar[i+1])

            for j, (u, v) in enumerate(P):
                Pc_bar[j].append(min(u.value[list(B)]) - min(v.value[list(B)]))

    if env is None:
        env = gp.Env(empty = True)
        env.setParam('OutputFlag', 0)
        env.start()
    m = gp.Model("Pairwise max regret choquet", env=env)
    w = m.addVars(len(subsets), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="moebius_masses")
    m.update()
    m.setObjective(gp.quicksum(w[i] * c_bar[i] for i in range(len(subsets))), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(w) == 1.0, name="sum_constraint")
    m.addConstr(w[0] == 0.0, name="empty_set_constraint")
    for index_subset, A in enumerate(subsets):
        for i in A:
            m.addConstr(gp.quicksum(w[index_subsubset] for index_subsubset, B in enumerate(subsets[:index_subset]) if B.issubset(A) and i in B) >= 0, name="monotonocity_constraints")         
    for pc_bar in Pc_bar:
        m.addConstr(gp.quicksum(w[i] * pc_bar[i] for i in range(len(subsets))) >= 0.0, name="preference_constraints")

    m.optimize()

    if m.status == GRB.INFEASIBLE:
        return float("-inf"), None
    # print("Pairwise max regret weighted sum: {} in {}s".format(m.ObjVal, time.time()-start))
    return [w[i].x for i in range(len(subsets))], m.ObjVal

def max_regret(x: DPoint, Y: list[DPoint], P: np.ndarray = [], pref_model: str = "ws", env: gp.Env = None) -> tuple[DPoint, float]:
    """
    Computes the max regret.
    :param x: the point
    :param Y: the set of points
    :param P: the set of known preferences
    :param env: the Gurobi environment
    :return: the max regret
    """
    xmr, mar = None, float("-inf")
    i = 0
    for y in Y:
        i += 1
        if x == y:
            continue
        # print("Max regret weighted sum: {}/{}".format(i, len(Y)))
        if pref_model == "ws":
            _, mar_ = pairwise_max_regret_ws(x, y, P, env)
        elif pref_model == "owa":
            _, mar_ = pairwise_max_regret_owa(x, y, P, env)
        elif pref_model == "choquet":
            _, mar_ = pairwise_max_regret_choquet(x, y, P, env)
        if mar_ > mar:
            xmr, mar = y, mar_

    return xmr, mar


def minimax_regret(X: list[DPoint], P: np.ndarray = [], pref_model: str = "ws", env: gp.Env = None) -> tuple[DPoint, float]:
    """
    Computes the minimax regret.
    :param X: the set of points
    :param P: the set of known preferences
    :param env: the Gurobi environment
    :return: the minimax regret
    """
    xmmar, mmar = None, float("inf")
    i = 0
    for x in X:
        i += 1
        print("Minimax regret {}: {}/{}".format(pref_model, i, len(X)), end="\r")
        _, mar = max_regret(x, X, P, pref_model, env)
        if mar < mmar:
            xmmar, mmar = x, mar

    print('\nMMR: {}'.format(mmar))
    return xmmar, mmar

def current_solution_strategy(X: list[DPoint], dm: [np.ndarray | Capacity], pref_model: str = "ws", env: gp.Env = None) -> tuple[DPoint, int, list[float]]:
    """
    Computes the optimal solution according to the current solution strategy.
    :param X: the set of points
    :param dm_weights: the weights of the decision maker, unknown by the algorithm only used to simulate the decision maker
    :param pref_model: the preference model, either "ws", "owa" or "choquet"
    :param env: the Gurobi environment
    :return: the optimal solution according to the current solution strategy
    """
    if pref_model == "ws":
        ev = lambda x: x.weighted_sum(dm)
    elif pref_model == "owa":
        ev = lambda x: x.owa(dm)
    elif pref_model == "choquet":
        ev = lambda x: x.choquet(dm)
    question_counter = 0
    P = []
    xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
    yp, _ = max_regret(xp, X, P, pref_model=pref_model, env=env)
    mmr_history = [mmr]
    while mmr > 0 and question_counter < MAX_QUESTIONS:
        question_counter += 1
        print("Question {}: {} vs {}".format(question_counter, xp, yp))
        if ev(xp) > ev(yp):
            P.append((xp, yp))
            X.remove(yp)
        else:
            P.append((yp, xp))
            X.remove(xp)

        xp, mmr = minimax_regret(X, P, pref_model=pref_model, env=env)
        yp, _ = max_regret(xp, X, P, pref_model=pref_model, env=env)
        mmr_history.append(mmr)

    return xp, question_counter, mmr_history