import numpy as np

from os.path import join
from os import listdir

from itertools import groupby

DATA_ROOT_PATH = join("..", "data")

def load_dataset(identifiers, dates):
	files_pairs = get_files_pairs(identifiers, dates)
	X = []
	y = []
	for points, scores in files_pairs:
		X.append(np.loadtxt(join(DATA_ROOT_PATH, points), delimiter=' '))
		y.append(np.loadtxt(join(DATA_ROOT_PATH, scores), delimiter=' ').reshape(-1,1))
	return np.vstack(X), np.vstack(y)

def get_files_pairs(identifiers, dates):
	files = [f.split('_') for f in listdir(DATA_ROOT_PATH)]
	files = [f for f in files if f[-3] in identifiers and f[-2] in dates]
	points = [f for f in files if f[0] == "point"]
	scores = [["score"] + f[1:] for f in points if ["score"] + f[1:] in files]
	return [("_".join(p), "_".join(s)) for (p,s) in zip(points, scores)]
	

if __name__ == '__main__':
	identifiers = ["bayes"]
	dates = ["12-12-17"]
	X, y = load_dataset(identifiers, dates)
	print(X.shape)
	print(y.shape)
	
