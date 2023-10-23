from typing import List
from dKP import DKPPoint

class NDList:
	"""
	ND-List data structure is a list of points that are non dominated with respect to each other.
	"""
	def	__init__(self, d: int, points: List[DKPPoint] = []) -> None:
		"""
		Constructor of the ND-List class.
		:param d: the dimension of the problem
		:param points: the list of points
		"""
		self.d = d
		self.points = points
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the ND-List.
		:return: a string representation of the ND-List
		"""
		return "NDList(d={}, |points|={})".format(self.d, len(self.points))
	
	def update(self, y: DKPPoint, verbose = False) -> bool:
		"""
		Updates the ND-List with a new point.
		:param y: the new point
		:param verbose: True if the procedure should be verbose, False otherwise
		:return: True if the point is accepted, False otherwise
		"""
		if verbose:
			print("-----Updating ND list with point {}".format(y))
		if len(self.points) == 0:
			if verbose:
				print("ND list is empty, insertion")
			self.points.append(y)
		else:
			to_be_removed = []
			for i in range(len(self.points)):
				if self.points[i].covers(y):
					if verbose:
						print("Point {} is covered by point {}, rejection".format(y, self.points[i]))
					return False
				if y.dominates(self.points[i]):
					if verbose:
						print("Point {} dominate point {}, deletion".format(y, self.points[i]))
					to_be_removed.append(self.points[i])
			for z in to_be_removed:
				self.points.remove(z)
			self.points.append(y)
		# assert self.is_pareto_front() # DEBUG
		return True

	def copy(self) -> 'NDList':
		"""
		Copies the ND-List.
		:return: the copy of the ND-List
		"""
		return NDList(self.d, self.points.copy())
	
	def get_pareto_front(self) -> List[DKPPoint]:
		"""
		Gets the pareto front of the ND-List.
		:return: the pareto front of the ND-List
		"""
		return self.points.copy()
	
	def is_pareto_front(self) -> bool:
		"""
		Checks if the current ND-List is a Pareto front.
		:return: True if the current ND-List is a Pareto front, False otherwise
		"""
		correct = True
		if self.points is not []:
			for i,x in enumerate(self.points[:-1]):
				for j,y in enumerate(self.points[i+1:]):
					if i != j:
						if x.covers(y):
							print("Inconsistency: {} covers {}".format(x,y))
							correct = False
						if y.covers(x):
							print("Inconsistency: {} covers {}".format(y,x))
							correct = False
		return correct