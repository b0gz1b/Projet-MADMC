from dKP import DKP
from NDList import NDList
from PLS import *
import time


if __name__ == '__main__':
	# Read the instance from a file
	dkp = DKP.from_file("data/2KP200-TA-0.dat").subinstance(150)
	# Print the instance
	print(dkp)

	m = 50

	# Test the PLS algorithm
	pop_in = P0(dkp, m=m, verbose = False)
	print(pop_in.tree_form_representation())
	start = time.time()
	pls_res = PLS(dkp, m=m, verbose = 1, struct="NDTree")
	end = time.time()
	print("Size of the pareto front: {}".format(len(pls_res)))
	assert NDList(dkp.d, pls_res).is_pareto_front()
	print("Found in: {:.2f} s".format(end-start))
