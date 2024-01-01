from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from Capacity import Capacity
from DKP import DKP
from DKPPoint import DKPPoint

M = 10000

def opt_ws(dKP: DKP, weights: list[float], env: gp.Env = None) -> tuple[float, DKPPoint]:
    """
    Computes the optimal value of the problem with respect to the weighted sum.
    :param weights: the weights
    :return: the optimal value of the problem with respect to the weighted sum
    """
    if env is None:
        env = gp.Env(empty = True)
        env.setParam('OutputFlag', 0)
        env.start()
    m = gp.Model("dKP ws", env=env)
    x = m.addMVar(shape=dKP.n, vtype=GRB.BINARY, name="x")
    # Set objective, maximize the value of the knapsack (sum of the values of the items in the knapsack) ponderated by the weights
    obj = gp.LinExpr()
    for i in range(dKP.n):
        for j in range(dKP.d):
            obj += dKP.v[i][j] * weights[j] * x[i]
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(x @ dKP.w <= dKP.W, name="capacity_constraint")
    m.update()
    m.optimize()
    return m.ObjVal, DKPPoint(dKP, x.X)
	
def opt_owa(dKP: DKP, weights: list[float], env: gp.Env = None) -> tuple[float, DKPPoint]:
    """
    Computes the optimal value of the problem with respect to the ordered weighted average.
    :param weights: the weights
    :param env: the Gurobi environment
    :return: the optimal value of the problem with respect to the ordered weighted average
    """
    if env is None:
        env = gp.Env(empty = True)
        env.setParam('OutputFlag', 0)
        env.start()
    m = gp.Model("dKP owa", env=env)
    y = m.addVars(dKP.d, vtype=GRB.CONTINUOUS, name="y")
    s = m.addVars(dKP.n, vtype=GRB.BINARY, name="s")
    b = m.addMVar(shape=(dKP.d, dKP.d), vtype=GRB.BINARY, name="b")
    m.update()
    m.setObjective(gp.quicksum(y[i] * weights[i] for i in range(dKP.d)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(s[i] * dKP.w[i] for i in range(dKP.n)) <= dKP.W, name="capacity_constraint")
    m.addConstrs((y[k] <= gp.quicksum(s[j]*dKP.v[j,i] for j in range(dKP.n)) + M * b[i,k] for i in range(dKP.d) for k in range(dKP.d)), name="PI_constraints_1")
    m.addConstrs((gp.quicksum(b[i,k] for i in range(dKP.d)) == k for k in range(dKP.d)), name="PI_constraints_2")

    m.optimize()
    return m.ObjVal, DKPPoint(dKP, [s[i].X for i in range(dKP.n)])

def opt_choquet(dKP: DKP, cap: Capacity, env: gp.Env = None) -> tuple[float, DKPPoint]:
    """
    Computes the optimal value of the problem with respect to the Choquet integral.
    :param cap: the capacity
    :param env: the Gurobi environment
    :return: the optimal value of the problem
    """
    if env is None:
        env = gp.Env(empty = True)
        env.setParam('OutputFlag', 0)
        env.start()
    m = gp.Model("dKP choquet", env=env)
    allsubsets = []
    for i in range(dKP.d+1):
        allsubsets.extend(combinations(range(dKP.d), i))
    y = m.addVars(dKP.d, vtype=GRB.CONTINUOUS, name="y")
    s = m.addVars(dKP.n, vtype=GRB.BINARY, name="s")
    l = m.addVars(2**dKP.d, vtype=GRB.CONTINUOUS, name="l")
    m.update()
    
    m.setObjective(gp.quicksum(cap.v(subset) * l[i] for i, subset in enumerate(allsubsets)), GRB.MAXIMIZE)
    
    m.addConstr(gp.quicksum(s[i] * dKP.w[i] for i in range(dKP.n)) <= dKP.W, name="capacity_constraint")
    m.addConstrs((y[i] == gp.quicksum(s[j]*dKP.v[j,i] for j in range(dKP.n)) for i in range(dKP.d)), name="y_constraints")
    for i in range(dKP.d):
        where_i = []
        for pos, sub in enumerate(allsubsets):
            if i in sub:
                where_i.append(pos)
        m.addConstr(gp.quicksum(l[pos] for pos in where_i) - y[i] <= 1)
    
    m.optimize()

    return m.ObjVal, DKPPoint(dKP, [s[i].X for i in range(dKP.n)])

def opt_decision_maker(dKP: DKP, dm, pref_model: str = "ws", env: gp.Env = None) -> tuple[float, DKPPoint]:
    """
    Computes the optimal value of the problem.
    :param dm: the weights or the capacity
    :param pref_model: the preference model, either "ws", "owa" or "choquet"
    :param env: the Gurobi environment
    :return: the optimal value of the problem
    """
    if pref_model == "ws":
        return opt_ws(dKP, dm, env)
    elif pref_model == "owa":
        return opt_owa(dKP, dm, env)
    elif pref_model == "choquet":
        return opt_choquet(dKP, dm, env)
    else:
        raise Exception("Unknown preference model: {}".format(pref_model))