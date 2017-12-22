import numpy as np

from os.path import join, abspath, dirname
from os import listdir

DATA_ROOT_PATH = join(dirname(abspath(__file__)), "..", "data")

def load_dataset(identifiers, dates, batches, target="score"):
	files_pairs = get_file_pairs(identifiers, dates, batches, target)
	X = []
	y = []
	for points, scores in files_pairs:
		X.append(np.loadtxt(join(DATA_ROOT_PATH, points), delimiter=' '))
		y.append(np.loadtxt(join(DATA_ROOT_PATH, scores), delimiter=' ').reshape(-1,1))
	return np.vstack(X), np.vstack(y)

def get_file_pairs(identifiers, dates, batches, target):
	files = [f.split('_') for f in listdir(DATA_ROOT_PATH)]
	files = [f for f in files if f[-3] in identifiers and f[-2] in dates and f[-1] in batches]
	files = sorted(files, key=lambda x: x[4])
	points = [f for f in files if f[0] == "point"]
	scores = [[target] + f[1:] for f in points if [target] + f[1:] in files]
	return [("_".join(p), "_".join(s)) for (p,s) in zip(points, scores)]
	
def binary_class(y, threshold=0.5):
	new_y = y.copy()
	new_y[np.where(y >= threshold)] = 1
	new_y[np.where(y < threshold)] = 0
	return new_y

	
