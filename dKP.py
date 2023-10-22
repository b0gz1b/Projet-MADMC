import numpy as np
from typing import List

class InvalidFileFormatError(Exception):
    """
	Raised when the file is not well formatted.
	"""
    def __init__(self):
        super().__init__("First item line should not appear before both the 'n <number of items>' and 'c w v1 ... v<d> ' lines.")


class DKP:
	"""
	dKP class is used to store the data of a d-dimensional knapsack problem.
	"""
	def __init__(self, d: int, n: int, W: int, w: np.ndarray, v: np.ndarray) -> None:
		"""
		Constructor of the dKP class.
		:param d: the dimension of the problem
		:param n: the number of items
		:param W: the capacity of the knapsack
		:param w: the weights of the items
		:param v: the values of the items
		"""
		self.d = d
		self.n = n
		self.W = W
		self.w = w
		self.v = v
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the dKP instance.
		:return: a string representation of the dKP instance
		"""
		return "dKP(d={}, n={})".format(self.d, self.n)

	@classmethod
	def from_file(cls, filename: str) -> 'DKP':
		"""
		Reads a dKP instance from a file.
		:param filename: the name of the file
		:return: a dKP instance
		"""
		i = 0
		with open(filename, "r") as f:
			for line in f:
				if line[0] == "i":
					data = line.split()
					try:
						w[i] = int(data[1])
						for j in range(d):
							v[i][j] = int(data[j + 2])
						i += 1
					except NameError:
						raise InvalidFileFormatError
				else:
					if line[0] == "W":
						data = line.split()
						W = int(data[1])
					elif line[0]=="n":
						data = line.split()	
						n=int(data[1])
						w = np.zeros(n, dtype=int)
					elif line[0] == "c":
						data = line.split()
						if len(data) > 1 and data[1] == "w":
							d = len(data) - 2
							v = np.zeros((n, d), dtype=int)
		f.close()
		return cls(d, n, W, w, v)
	
	def generate_random_solution(self) -> List[int]:
		"""
		Generates a random solution.
		:return: a random solution
		"""
		x = np.zeros(self.n, dtype=int)
		arr = np.arange(self.n)
		np.random.shuffle(arr)
		wTotal = 0
		for i in range(self.n):
			if wTotal + self.w[arr[i]] <= self.W:
				x[arr[i]] = 1
				wTotal = wTotal + self.w[arr[i]]
		return x

class DPoint:
	"""
	Point data structure for any multi-objective problem.
	"""
	def __init__(self, value: np.ndarray) -> None:
		"""
		Constructor of the Point class.
		:param value: the value of the point
		"""
		self.value = value
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the Point instance.
		:return: a string representation of the Point instance
		"""
		return str(self.value)

	def dominates(self, other: 'DPoint') -> bool:
		"""
		Checks if the current point dominates another point.
		:param other: the other point
		:return: True if the current point dominates the other point, False otherwise
		"""
		return np.all(self.value >= other.value) and np.any(self.value > other.value)
	
	def covers(self, other: 'DPoint') -> bool:
		"""
		Checks if the current point covers another point.
		:param other: the other point
		:return: True if the current point covers the other point, False otherwise
		"""
		return self.dominates(other) or np.all(self.value == other.value)
	
	def euclidean_distance(self, other: 'DPoint') -> float:
		"""
		Computes the euclidean distance between the current point and another point.
		:param other: the other point
		:return: the euclidean distance between the current point and another point
		"""
		return np.linalg.norm(self.value - other.value)
	
	def average_euclidean_distance(self, others: List['DPoint']) -> float:
		"""
		Computes the average euclidean distance between the current point and a list of other points.
		:param others: the list of other points
		:return: the average euclidean distance between the current point and a list of other points
		"""
		return np.mean([self.euclidean_distance(other) for other in others])

class DKPPoint(DPoint):
	"""
	Point data structure for the dKP problem.
	"""
	def __init__(self, dkp: DKP, x: List[int] = [], weight: int = None, value: np.ndarray = None) -> None:
		"""
		Constructor of the Point class.
		:param dkp: the dKP instance
		:param x: the list of items in the knapsack, where x[i] = 1 if item i is in the knapsack, 0 otherwise
		"""
		super().__init__(np.dot(dkp.v.T, x) if value is None else value)
		self.dkp = dkp
		self.x = x
		self.weight = np.dot(dkp.w, x) if weight is None else weight
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the Point instance.
		:return: a string representation of the Point instance
		"""
		return str(self.value)

	def neighbors_one_one(self) -> List['DKPPoint']:
		"""
		Computes the list of neighbors of the point that can be obtained by removing and adding one item from the knapsack.
		:return: the list of neighbors of the point that can be obtained by removing and adding one item from the knapsack
		"""
		neighbors = []
		for i in range(self.dkp.n):
			for j in range(self.dkp.d):
				if self.x[i] == 1 and self.x[j] == 0:
					weight = self.weight - self.dkp.w[i] + self.dkp.w[j]
					if weight <= self.dkp.W:
						x = self.x.copy()
						x[i] = 0
						x[j] = 1
						neighbors.append(DKPPoint(self.dkp, x, weight = weight, value = self.value - self.dkp.v[i] + self.dkp.v[j]))
		return neighbors
	

class NDTreeNode:
	"""
	ND-Tree node data structure.
	"""
	def __init__(self, 
			  tree: 'NDTree',
			  S: List[DKPPoint] = [], 
			  L: List[DKPPoint] = [], 
			  parent: 'NDTreeNode' = None, 
			  children: List['NDTreeNode'] = []
			  ) -> None:
		"""
		Constructor of the ND-Tree node class.
		:param tree: the ND-Tree
		:param S: the set of points associated with the node
		:param L: the list of points associated with the node if it is a leaf
		:param parent: the parent node
		:param children: the list of children nodes
		"""
		self.tree = tree
		self.children = children
		self.parent = parent
		self.S = S
		self.L = L
		self.ideal = DPoint(np.full(tree.d, -np.inf))
		self.nadir = DPoint(np.full(tree.d, np.inf))
		for s in S:
			self.ideal.value = np.maximum(self.ideal.value, s.value)
			self.nadir.value = np.minimum(self.nadir.value, s.value)
		
	def __str__(self) -> str:
		"""
		Gives a string representation of the ND-Tree node.
		:return: a string representation of the ND-Tree node
		"""
		return "NDTreeNode({}, |S|={}, |children|={}, ideal={}, nadir={})".format("leaf" if self.is_leaf() else "internal",len(self.S),len(self.children), self.ideal, self.nadir)

	def is_leaf(self) -> bool:
		"""
		Checks if the node is a leaf.
		:return: True if the node is a leaf, False otherwise
		"""
		return len(self.children) == 0
	
	def update_node(self, y: DKPPoint, verbose = False) -> bool:
		"""
		Checks if the new point y is covered or non dominated with respect to S(n) by the following procedure:
			Compare the new point y to the approximate ideal point and nadir point of the current node. 
			If the new point is covered by the approximate nadir point, then the point is rejected.
			If the new point covers the approximate ideal point, then the node is deleted.
			Otherwise, if y is covered by the approximate ideal point or covers the approximate nadir point, then the node is further analyzed.
			If the current node is internal, all its children are updated.
			If the current node is a leaf, then the new point is compared to every point in L(n).
			If y is dominated by a point in L(n), then y is rejected, else if y dominates a point in L(n), then the dominated point is deleted from L(n).
		:param y: the new point
		:return: True if the point is accepted, False otherwise
		"""
		if verbose:
			print("Updating node {} with point {}".format(self, y))
		if self.nadir.covers(y):
			# Property 1, y is covered by the approximate nadir point so it is rejected
			if verbose:
				print("Point {} is covered by the approximate nadir point {}".format(y, self.nadir))
			return False
		elif y.covers(self.ideal):
			# Property 2, y covers the approximate ideal point so the node is deleted
			if verbose:
				print("Point {} covers the approximate ideal point {}".format(y, self.ideal))
			self.prune_yourself()
		elif self.ideal.covers(y) or y.covers(self.nadir):
			if verbose:
				print("Point {} is covered by the approximate ideal point {} or covers the approximate nadir point {}".format(y, self.ideal, self.nadir))
			if self.is_leaf():
				print("Case : Leaf node")
				to_be_pruned = []
				for z in self.L:
					if z.covers(y):
						# y is covered by a point in L(n) so it is rejected
						if verbose:
							print("Point {} is covered by point {}, rejection".format(y, z))
						return False
					elif y.dominates(z):
						# y dominates a point in L(n) so the dominated point is deleted from L(n)
						if verbose:
							print("Point {} dominates point {}, deletion".format(y, z))
						to_be_pruned.append(z)
						self.L.remove(z)
				# we pass on the list of points to be pruned to the parent node too
				if verbose:
					print("Pruning points: {}".format([str(tbp) for tbp in to_be_pruned]))
				self.update_S_pruning(to_be_pruned)
			else:
				if verbose:
					print("Case : Internal node")
				removed_children = []
				for child in self.children:
					if verbose:
						print("Updating child node {}".format(child))
					if not child.update_node(y, verbose=verbose):
						# y is rejected by a child node so it is rejected
						if verbose:
							print("Point {} is rejected by child node {}".format(y, child))
						return False
					else:
						if child.parent == None:
							if verbose:
								print("Child node {} is pruned".format(child))
							# the child node was deleted so we delete it from the list of children
							removed_children.append(child)
				for child in removed_children:
					if verbose:
						print("Removing child node {} from parent node {}".format(child, self))
					self.children.remove(child)
				if len(self.children) == 1:
					if verbose:
						print("Parent node {} has only one child node {} so it is replaced by this child node".format(self, self.children[0]))
					# the current node has only one child so it is replaced by this child
					self.children[0].parent = self.parent
					self.parent.children.remove(self)
					self.parent.children.append(self.children[0])
		else:
			# Property 3, y is non dominated with respect to the approximate ideal point and approximate nadir point so y is non dominated with respect to S(n)
			if verbose:
				print("Point {} is non dominated with respect to the approximate ideal point {} and approximate nadir point {}, node is skipped".format(y, self.ideal, self.nadir))
			pass # We can skip this node
		return True
	
	def insert(self, y: DKPPoint) -> None:
		"""
		Inserts a new point in the ND-Tree.
		:param y: the new point
		"""
		if self.is_leaf():
			self.L.append(y)
			self.S.append(y)
			self.update_ideal_nadir(y)
			if len(self.L) > self.tree.max_leaf_size:
				self.split()
		else:
			self.S.append(y)


	def split(self) -> None:
		"""
		Splits the current node into number_of_children children using a simple clustering heuristic based on euclidean distance.
		"""
		children_points = []
		max_average_distance = 0
		farthest_point_index = 0
		# First we find the point that is the farthest from all the other points in L
		for i in range(len(self.L)):
			average_distance = self.L[i].average_euclidean_distance(self.L)
			if average_distance > max_average_distance:
				max_average_distance = average_distance
				farthest_point_index = i
		# Then we create a new child with this point
		self.children.append(NDTreeNode(self.tree, [self.L[farthest_point_index]], [self.L[farthest_point_index]], self))
		children_points.append(self.L.pop(farthest_point_index))
		# Then, and until we have number_of_children children
		while len(self.children) < self.tree.number_of_children:
			max_average_distance = 0
			farthest_point_index = 0
			# we find the point that is the farthest from all the other points in children of the current node
			for i in range(len(self.L)):
				average_distance = self.L[i].average_euclidean_distance(children_points)
				if average_distance > max_average_distance:
					max_average_distance = average_distance
					farthest_point_index = i
			# and we create a new child with this point
			self.children.append(NDTreeNode(self.tree, [self.L[farthest_point_index]], [self.L[farthest_point_index]], self))
			children_points.append(self.L.pop(farthest_point_index))
		# Finally, we add the remaining points in L to their closest child
		while len(self.L) > 0:
			z = self.L.pop()
			closest_child_index = self.find_closest_child_index(z)
			self.children[closest_child_index].S.append(z)
			self.children[closest_child_index].L.append(z)
			self.children[closest_child_index].update_ideal_nadir(z)

	def update_ideal_nadir(self, y: DKPPoint) -> None:
		"""
		Updates the approximate ideal point and approximate nadir point of the current node and its parents.
		:param y: the new point
		"""
		new_ideal = np.maximum(self.ideal.value, y.value)
		new_nadir = np.minimum(self.nadir.value, y.value)
		if np.any(new_ideal != self.ideal.value) or np.any(new_nadir != self.nadir.value):
			self.ideal.value = new_ideal
			self.nadir.value = new_nadir
			if self.parent is not None:
				self.parent.update_ideal_nadir(y)

	def find_closest_child_index(self, y: DKPPoint) -> int:
		"""
		Finds the closest child of the current node with respect to a new point.
		:param y: the new point
		:return: the index of the closest child
		"""
		min_distance = 0
		closest_child_index = 0
		for i in range(len(self.children)):
			distance = y.euclidean_distance(self.children[i].get_midde_point())
			if distance < min_distance:
				min_distance = distance
				closest_child_index = i
		return closest_child_index

	def prune_yourself(self) -> None:
		"""
		Prunes the current node.
		"""
		if self.parent is not None:
			self.parent.prune_your_child(self)
			self.parent = None
	
	def prune_your_child(self, child: 'NDTreeNode') -> None:
		"""
		Prunes a child of the current node.
		:param child: the child to be pruned
		"""
		self.update_S_pruning(child.S)

	def update_S_pruning(self, to_be_pruned: List[DKPPoint]) -> None:
		"""
		Updates the set of points associated with the current node.
		:param to_be_pruned: the list of points to be pruned
		"""
		self.S = [s for s in self.S if s not in to_be_pruned]
		if self.parent is not None:
			self.parent.update_S_pruning(to_be_pruned)
	
	def	get_midde_point(self) -> DPoint:
		"""
		Computes the middle point of the current node.
		:return: the middle point of the current node
		"""
		return DPoint((self.ideal.value + self.nadir.value) / 2)


class NDTree:
	"""
	ND-Tree data structure is a tree with the following properties :
		1) With each node n is associated a set of points S(n).
		2) Each leaf node contains a list L(n) of points and S(n) = L(n).
		3) For each internal node n, S(n) is the union of disjoint sets associated with all children of n.
		4) Each node n stores an approximate ideal point ideal_hat(S(n)) and approximate nadir point nadir_hat(S(n)).
		5) If n' is a child of n, then ideal_hat(S(n)) covers ideal_hat(S(n')) and nadir_hat(S(n')) covers nadir_hat(S(n)).
	[A. Jaszkiewicz and T. Lust, "ND-Tree-Based Update: A Fast Algorithm for the Dynamic Nondominance Problem," in IEEE Transactions on Evolutionary Computation, vol. 22, no. 5, pp. 778-791, Oct. 2018, doi: 10.1109/TEVC.2018.2799684.]
	"""
	def __init__(self, d: int, number_of_children: int, max_leaf_size: int) -> None:
		"""
		Constructor of the ND-Tree class.
		:param d: the dimension of the problem
		:param number_of_children: the number of children of each node
		:param max_leaf_size: the maximum size of the leaf nodes
		"""
		self.d = d
		self.number_of_children = number_of_children
		self.max_leaf_size = max_leaf_size
		self.root = None
	
	def update(self, y: DKPPoint, verbose = False) -> None:
		"""
		Updates the ND-Tree with a new point.
		:param y: the new point
		"""
		if self.root is None:
			if verbose:
				print("Creating root node")
			self.root = NDTreeNode(self, [y], [y])
		elif self.root.update_node(y, verbose=verbose):
			if verbose:
				print("Inserting point {} in ND tree".format(y))
			self.root.insert(y)