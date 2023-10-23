from dKP import *
from PLS import *
import numpy as np

if __name__ == '__main__':
	# Read the instance from a file
	dkp = DKP.from_file("data/2KP200-TA-0.dat")
	# Print the instance
	print(dkp)
	# Generate m random solutions
	m = 1000
	solutions = []
	for i in range(m):
		solution = DKPPoint(dkp, dkp.generate_random_solution())
		solutions.append(solution)
	
	number_of_children = dkp.d + 1
	max_leaf_size = 20
	# Create the pareto front archive
	Yn = NDTree(dkp.d, number_of_children, max_leaf_size)
	# Update the pareto front of the archive
	for i in range(m):
		Yn.update(solutions[i], verbose=False)
	
	# Print the tree form representation
	print(Yn.tree_form_representation())
	
	assert Yn.is_pareto_front()


	# Create the pareto front archive using a list
	YnList = NDList(dkp.d)
	# Update the pareto front of the archive
	for i in range(m):
		YnList.update(solutions[i], verbose=False)

	assert YnList.is_pareto_front()
	assert all([sol in YnList.points for sol in Yn.root.S]) and all([sol in Yn.root.S for sol in YnList.points])
	# Test the copy method
	Yncp = Yn.copy()

	# Test the PLS algorithm
	PLS(dkp, m)