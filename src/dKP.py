import numpy as np
from uuid import uuid4

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
						raise InvalidFileFormatError()
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
	
	def __str__(self) -> str:
		"""
		Gives a string representation of the dKP instance.
		:return: a string representation of the dKP instance
		"""
		return "dKP(d={}, n={}, W={})".format(self.d, self.n, self.W)

	
	def subinstance(self, n: int, d: int, save: str = "", shuffle: bool = False) -> 'DKP':
		"""
		Computes a subinstance of the dKP instance, with n items randomly selected for d randomly selected values.
		:param n: the number of items of the subinstance
		:param d: the dimension of the subinstance
		:param save: the path to save the subinstance
		:param shuffle: whether to shuffle the items or not
		:return: a subinstance of the dKP instance
		"""
		arr = np.arange(self.n)
		if shuffle:
			np.random.shuffle(arr)
		arr2 = np.arange(self.d)
		if shuffle:
			np.random.shuffle(arr2)
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

	
	def generate_random_solution(self) -> list[int]:
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
	
	def R_i(self, q: list[float], i: int) -> float:
		"""
		Computes the performance ratio of the item i with respect to the ponderation vector q.
		:param q: the ponderation vector
		:param i: the index of the item
		:return: the performance ratio of the item i with respect to the ponderation vector q
		"""
		return np.dot(q, self.v[i]) / self.w[i]
	
	def R(self, q: list[float]) -> float:
		"""
		Computes the performance ratio of all the items with respect to the ponderation vector q.
		:param q: the ponderation vector
		:return: the performance ratio of all the items with respect to the ponderation vector q
		"""
		return np.dot(q, self.v.T) / self.w

class InvalidFileFormatError(Exception):
	"""
	Raised when the file is not well formatted.
	"""
	def __init__(self) -> None:
		"""
		Constructor of the InvalidFileFormatError class.
		"""
		super().__init__("First item line should not appear before both the 'n <number of items>' and 'c w v1 ... v<d> ' lines.")