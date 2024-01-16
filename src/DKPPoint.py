import numpy as np
from DPoint import DPoint
from DKP import DKP
from typing import List

class DKPPoint(DPoint):
	"""
	Point data structure for the dKP problem.
	"""
	def __init__(self, dkp: 'DKP', x: List[int] = [], weight: int = None, value: np.ndarray = None) -> None:
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