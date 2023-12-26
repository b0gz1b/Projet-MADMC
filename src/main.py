from dKP import DKP
from NDList import NDList
from PLS import *
import time

N = 35
D = 5

if __name__ == '__main__':
	# Read the instance from a file
	dkp = DKP.from_file("data/2KP200-TA-0.dat")

	sub_dkp = dkp.subinstance(N, D)

	# Print the instance
	print(sub_dkp)

	m = 100

	# Test the PLS algorithm
	pop_in = P0(sub_dkp, m=m, verbose = False)
	print(pop_in.tree_form_representation())
	for struct in ["NDTree", "NDList"]:
		start = time.time()
		print("Using {}".format(struct))
		pls_res = PLS(sub_dkp, m=m, verbose = 1, struct=struct, initial_pop=pop_in)
		end = time.time()
		print("Size of the pareto front: {}".format(len(pls_res)))
		assert NDList(sub_dkp.d, pls_res).is_pareto_front()
		print("Found in: {:.2f} s".format(end-start))
