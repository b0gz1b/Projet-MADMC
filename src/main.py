from DKP import DKP
from PLS import *
import elicitation as eli
import utils as ut
import PL_optimal as opt
import gurobipy as gp
from time import time

if __name__ == '__main__':
	# Read the instance from a file
	dkp = DKP.from_file("data/2KP200-TA-0.dat")
	N = 20
	D = 3
	sub_dkp = dkp.subinstance(N, D, shuffle=True)
	# Simulate the decision maker
	dm = ut.generate_convex_capacity_choquet(D, 1)[0]
	# Compute the optimal solution
	true_opt, sol = opt.opt_choquet(sub_dkp, dm)
	sol = DKPPoint(sub_dkp, sol)
	print("True optimal solution: {}".format(true_opt))
	# sol is a list of 0 and 1, 1 if the item is in the knapsack, 0 otherwise
	# compute the point corresponding to sol
	print("True optimal point: {}".format(sol))
	m = 1

	pop_in = P0(sub_dkp, m=m, verbose = False)
	X = PLS(sub_dkp, m=m, verbose = 0, struct = "NDTree", initial_pop = pop_in)
	print("Pareto front size: {}".format(len(X)))
	

	# Compute the current solution strategy
	env = gp.Env(empty = True)
	env.setParam('OutputFlag', 0)
	env.start()
	start = time()
	xopt, nb_questions, mmr_hist = eli.current_solution_strategy(X, dm, pref_model="choquet", env = env)
	end = time()
	print("Current solution strategy: {} with {} questions in {:.2f}s".format(xopt, nb_questions, end-start))
	
	
	# Compute the error
	error = sol.choquet(dm) - xopt.choquet(dm)
	print("Error: {}".format(error))