import numpy as np
import matplotlib.pyplot as plt

from dataset import load_dataset, binary_class
from classification import classification_experiment
from dim_reduction import explained_variance, apply_PCA

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def plot_score(y, savefig=None):
	plt.plot(y[np.where(y != 0)])
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel("Experiment iteration", fontsize=25)
	plt.ylabel("Profile score", fontsize=25)
	if savefig is None:
		plt.show()
	else:
		plt.savefig(savefig)
		plt.close()

def main():
	random_state = 42
	test_size = 0.25
	n_folds = 5
	np.random.seed(random_state)
	
	identifiers = ["bayes",  "adhoc"]
	dates = ["12-12-17"]
	batches = ["1","2","3"]
	
	for identifier in identifiers:
		print("--- " + identifier + " ---")

		X, y = load_dataset(identifier, dates, batches, "score")
		X, y_label = load_dataset(identifier, dates, batches, "label")
		y_label = y_label.ravel()
		
		plot_score(y, savefig=identifier + "_timewise.pdf")
		
		print("Classes distribution: ")
		print(np.unique(y_label, return_counts=True))
		classification_experiment(X, y_label, test_size, n_folds, random_state, savefig=identifier)
		
		explained_variance(X,y, savefig=identifier + "_explained_var.pdf")
		
		# Selecting only class with a single spot
		index = np.where(y_label ==1)
		X = X[index]
		y = y[index]
		y_label = y_label[index]
		
		apply_PCA(X, y, savefig=identifier + "_single_PCA.pdf")
		
	print("### DONE ###")

if __name__ == '__main__':
	main()
