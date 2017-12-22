import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
	
def classification(X, y, results, test_size, n_folds, random_state):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=random_state)

	# #############################################################################
	# Decision Tree
	
	tree = DecisionTreeClassifier()
	tree_params = [{'criterion': ['gini', 'entropy'], 'max_depth':[3, 5, 7, 9, None]}]
	clf_tree = GridSearchCV(tree, tree_params, cv=n_folds, refit=True, n_jobs=4).fit(X_train, y_train)
	y_pred =  clf_tree.predict(X_test)
	results['tree']['train'].append(accuracy_score(y_train, clf_tree.predict(X_train)))
	results['tree']['test'].append(accuracy_score(y_test, y_pred))
	results['tree']['confusion'].append(confusion_matrix(y_test, y_pred))

	# #############################################################################
	# Random Forest
	
	forest = RandomForestClassifier(random_state=random_state)
	forest_params = [{'criterion': ['gini', 'entropy'], 'n_estimators':[10,20,30,40,50, 60]}]
	clf_forest = GridSearchCV(forest, forest_params, cv=n_folds, refit=True, n_jobs=4).fit(X_train, y_train)
	y_pred =  clf_forest.predict(X_test)
	results['forest']['train'].append(accuracy_score(y_train, clf_forest.predict(X_train)))
	results['forest']['test'].append(accuracy_score(y_test, y_pred))
	results['forest']['confusion'].append(confusion_matrix(y_test, y_pred))
	
	# #############################################################################
	# Adaboost
	
	ada = AdaBoostClassifier(random_state=random_state)
	ada_params = [{'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)], 'n_estimators':[50, 100, 150, 200]}]
	clf_ada = GridSearchCV(ada,ada_params, cv=n_folds, refit=True, n_jobs=4).fit(X_train, y_train)
	y_pred =  clf_ada.predict(X_test)
	results['ada']['train'].append(accuracy_score(y_train, clf_forest.predict(X_train)))
	results['ada']['test'].append(accuracy_score(y_test, y_pred))
	results['ada']['confusion'].append(confusion_matrix(y_test, y_pred))
	
def confusion_heatmap(cnf_matrices, clf, savefig):
	classes = ["No spot", "Single spot", "Multi-spots"]
	cnf_matrices = [cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis] for cnf_matrix in cnf_matrices]
	total = np.zeros((len(classes),len(classes)))
	for matrix in cnf_matrices:
		total = np.add(total, matrix)
	cnf_matrix = total/len(cnf_matrices)
	
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
	plt.yticks(tick_marks, classes, fontsize=15)

	thresh = cnf_matrix.mean()
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, format(cnf_matrix[i, j], '.2f'), horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black", fontsize=16)
		
	plt.tight_layout()
	if savefig is None:
		plt.show()
	else:
		plt.savefig(savefig + "_cnf_matrix_" + clf + ".pdf")
		plt.close()
	
def generate_results(results, clfs, metrics, savefig):
	for clf in clfs:
		print("\n--- " + clf+ " ---")
		for metric in metrics:
			if metric == 'confusion':
				confusion_heatmap(results[clf][metric], clf, savefig)
			else:
				print("\nMean accuracy on " + metric + ": {0:.3f}".format(np.mean(results[clf][metric])))
				print("Std accuracy on " + metric + ": {0:.3f}".format(np.std(results[clf][metric])))

def classification_experiment(X, y, test_size, n_folds, random_state, savefig=None):
	seeds = [40, 41, 42, 43, 44, 45, 56, 47, 48, 49]
	clfs = ['tree', 'forest', 'ada']
	metrics = ['train', 'test', 'confusion']
	results = {clf:{metric:[] for metric in metrics} for clf in clfs}
	
	for seed in seeds:
		classification(X, y, results, test_size, n_folds, seed)
	generate_results(results, clfs, metrics, savefig)
