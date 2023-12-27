import numpy as np
from typing import List
from uuid import uuid4
import gurobipy as gp
from gurobipy import GRB

class InvalidFileFormatError(Exception):
	"""
	Raised when the file is not well formatted.
	"""
	def __init__(self) -> None:
		"""
		Constructor of the InvalidFileFormatError class.
		"""
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
		return "dKP(d={}, n={}, W={})".format(self.d, self.n, self.W)

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
	
	def subinstance(self, n: int, d: int, save: str = "") -> 'DKP':
		"""
		Computes a subinstance of the dKP instance, with n items randomly selected for d randomly selected values.
		:param n: the number of items of the subinstance
		:param d: the dimension of the subinstance
		:param save: the path to save the subinstance
		:return: a subinstance of the dKP instance
		"""
		arr = np.arange(self.n)
		# np.random.shuffle(arr)
		arr2 = np.arange(self.d)
		# np.random.shuffle(arr2)
		new_w = self.w[arr[:n]]
		sub = DKP(d, n, sum(new_w) // 2, new_w, self.v[arr[:n]][:, arr2[:d]])
		if save != "":
			with open(save + "/{}KP{}S{}KP{}-TA-{}.dat".format(d, n, self.d, self.n, uuid4()), "w") as f:
				f.write("c Instance Type h\n")
				f.write("c")
				f.write("n {}\n".format(str(sub.n)))
				f.write("c w")
				for i in range(sub.d):
					f.write(" v{}".format(i + 1))
				f.write("\n")
				for i in range(sub.n):
					f.write("i {} ".format(str(sub.w[i])) + " ".join(map(str, sub.v[i])))
					f.write("\n")
				f.write("c\nc\n")
				f.write("W {}\n".format(str(sub.W)))
				f.write("c end of file")
			f.close()
		return sub

	
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
	
	def R_i(self, q: List[float], i: int) -> float:
		"""
		Computes the performance ratio of the item i with respect to the ponderation vector q.
		:param q: the ponderation vector
		:param i: the index of the item
		:return: the performance ratio of the item i with respect to the ponderation vector q
		"""
		return np.dot(q, self.v[i]) / self.w[i]
	
	def R(self, q: List[float]) -> float:
		"""
		Computes the performance ratio of all the items with respect to the ponderation vector q.
		:param q: the ponderation vector
		:return: the performance ratio of all the items with respect to the ponderation vector q
		"""
		return np.dot(q, self.v.T) / self.w
	
	def opt(self, weights: List[float], env: gp.Env = None) -> tuple[float, List[int]]:
		"""
		Computes the optimal value of the problem.
		:param weights: the weights
		:return: the optimal value of the problem
		"""
		if env is None:
			env = gp.Env(empty = True)
			env.setParam('OutputFlag', 0)
			env.start()
		m = gp.Model("dKP", env=env)
		x = m.addMVar(shape=self.n, vtype=GRB.BINARY, name="x")
		# Set objective, maximize the value of the knapsack (sum of the values of the items in the knapsack) ponderated by the weights
		obj = gp.LinExpr()
		for i in range(self.n):
			for j in range(self.d):
				obj += self.v[i][j] * weights[j] * x[i]
		m.setObjective(obj, GRB.MAXIMIZE)
		m.addConstr(x @ self.w <= self.W, name="capacity_constraint")
		m.update()
		m.optimize()
		return m.ObjVal, x.X
		


class DPoint:
	"""
	Point data structure for any multi-objective problem.
	"""
	def __init__(self, value: np.ndarray) -> None:
		"""
		Constructor of the Point class.
		:param value: the value of the point
		"""
		self.value = np.asarray(value)
		self.dimension = len(value)
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the Point instance.
		:return: a string representation of the Point instance
		"""
		return str(self.value)
	
	def __sub__(self, __value: 'DPoint') -> 'DPoint':
		"""
		Subtracts two points.
		:param __value: the other point
		:return: the difference between the two points
		"""
		return DPoint(self.value - __value.value)
	
	def __add__(self, __value: 'DPoint') -> 'DPoint':
		"""
		Adds two points.
		:param __value: the other point
		:return: the sum of the two points
		"""
		return DPoint(self.value + __value.value)
	
	def __mul__(self, __value: 'DPoint') -> 'DPoint':
		"""
		Multiplies two points.
		:param __value: the other point
		:return: the product of the two points
		"""
		return DPoint(self.value * __value.value)
	
	def __eq__(self, __value: 'DPoint') -> bool:
		"""
		Checks if the current point is equal to another point.
		:param __value: the other point
		:return: True if the current point is equal to the other point, False otherwise
		"""
		return np.all(self.value == __value.value)
	
	def __hash__(self) -> int:
		"""
		Computes the hash of the point.
		:return: the hash of the point
		"""
		return hash(tuple(self.value))

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
	
	def sum(self) -> float:
		"""
		Computes the sum of the values of the point.
		:return: the sum of the values of the point
		"""
		return np.sum(self.value)
	
	def weighted_sum(self, weights: np.ndarray) -> float:
		"""
		Computes the weighted sum of the values of the point.
		:param weights: the weights
		:return: the weighted sum of the values of the point
		"""
		if len(weights) != len(self.value):
			raise ValueError("The length of the weights vector should be equal to the length of the value vector.")
		return np.dot(self.value, weights)
	
	def owa(self, weights: np.ndarray) -> float:
		"""
		Computes the ordered weighted average of the values of the point.
		:param weights: the weights
		:return: the ordered weighted average of the values of the point
		"""
		if len(weights) != len(self.value):
			raise ValueError("The length of the weights vector should be equal to the length of the value vector.")
		return np.dot(np.sort(self.value), weights)
	
	def choquet(self, capacity: np.ndarray) -> float:
		"""
		Computes the Choquet integral of the values of the point.
		:param capacity: the capacity
		:return: the Choquet integral of the values of the point
		"""
		raise NotImplementedError("Choquet integral is not implemented yet.")

class DKPPoint(DPoint):
	"""
	Point data structure for the dKP problem.
	"""
	def __init__(self, dkp: DKP, x: List[int] = [], weight: int = None, value: np.ndarray = None) -> None:
		"""
		Constructor of the Point class.
		:param dkp: the dKP instance
		:param x: the list of items in the knapsack, where x[i] = 1 if item i is in the knapsack, 0 otherwise
		:param weight: the weight of the point
		:param value: the value of the point
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
		return np.array2string(self.value)

	def neighbors_one_one(self) -> List['DKPPoint']:
		"""
		Computes the list of neighbors of the point that can be obtained by removing and adding one item from the knapsack.
		:return: the list of neighbors of the point that can be obtained by removing and adding one item from the knapsack
		"""
		neighbors = []
		for i in range(self.dkp.n):
			for j in range(self.dkp.n):
				if self.x[i] == 1 and self.x[j] == 0:
					weight = self.weight - self.dkp.w[i] + self.dkp.w[j]
					if weight <= self.dkp.W:
						x = self.x.copy()
						x[i] = 0
						x[j] = 1
						neighbors.append(DKPPoint(self.dkp, x, weight = weight, value = self.value - self.dkp.v[i] + self.dkp.v[j]))
		return neighbors
	
	def get_items(self):
		"""
		Gets the items in the knapsack.
		:return: the items in the knapsack
		"""
		return np.where(self.x == 1)[0]
	