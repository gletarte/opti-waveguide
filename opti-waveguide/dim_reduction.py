import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection

def explained_variance(X, y, savefig=None):
	print("Computing explained variance")
	pca = decomposition.PCA()

	# Plot the PCA spectrum
	pca.fit(X)

	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_polynoms')
	plt.ylabel('explained_variance')
	if savefig is not None:
		plt.savefig(savefig)
		plt.close()
	else:
		plt.show()
	
def apply_MDS_embedding(X, y, y_label, random_state=42, savefig=None):
	#----------------------------------------------------------------------
	# MDS  embedding of the digits dataset
	print("Computing MDS embedding")
	clf = manifold.MDS(n_components=2, n_init=1, max_iter=100, n_jobs=-1, random_state=random_state)
	X_mds = clf.fit_transform(X)
	
	print("Done. Stress: %f" % clf.stress_)
	plot_3D_labeled_datapoints(X_mds, y, y_label,savefig)
	
def apply_PCA(X, y, y_label=None, random_state=42, savefig=None):
	#----------------------------------------------------------------------
	# Projection on to the first 2 principal components
	print("Computing PCA projection")
	X_pca = decomposition.PCA(n_components=2, random_state=random_state).fit_transform(X)
	if y_label is None:
		plot_3D_datapoints(X_pca, y, savefig)
	else:
		plot_3D_labeled_datapoints(X_pca, y, y_label, savefig)
	
def try_all_dim_reduction(X, y, y_label):
	n_neighbors = 30
	#----------------------------------------------------------------------
	# Random 2D projection using a random unitary matrix
	print("Computing random projection")
	rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
	X_projected = rp.fit_transform(X)
	plot_3D_labeled_datapoints(X_projected, y, y_label)


	#----------------------------------------------------------------------
	# Projection on to the first 2 principal components
	print("Computing PCA projection")
	X_pca = decomposition.PCA(n_components=2).fit_transform(X)
	plot_3D_labeled_datapoints(X_pca, y, y_label)


	#----------------------------------------------------------------------
	# Isomap projection of the digits dataset
	print("Computing Isomap embedding")
	X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
	plot_3D_labeled_datapoints(X_iso, y, y_label)


	#----------------------------------------------------------------------
	# Locally linear embedding of the digits dataset
	print("Computing LLE embedding")
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
										  method='standard')
	X_lle = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_3D_labeled_datapoints(X_lle,y, y_label)


	#----------------------------------------------------------------------
	# Modified Locally linear embedding of the digits dataset
	print("Computing modified LLE embedding")
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')

	X_mlle = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_3D_labeled_datapoints(X_mlle, y, y_label)

	#----------------------------------------------------------------------
	# MDS  embedding of the digits dataset
	print("Computing MDS embedding")
	clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)

	X_mds = clf.fit_transform(X)
	print("Done. Stress: %f" % clf.stress_)
	plot_3D_labeled_datapoints(X_mds, y, y_label)

	#----------------------------------------------------------------------
	# Random Trees embedding of the digits dataset
	print("Computing Totally Random Trees embedding")
	hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
										   max_depth=5)
	X_transformed = hasher.fit_transform(X)
	pca = decomposition.TruncatedSVD(n_components=2)
	X_reduced = pca.fit_transform(X_transformed)

	plot_3D_labeled_datapoints(X_reduced, y, y_label)

	#----------------------------------------------------------------------
	# Spectral embedding of the digits dataset
	print("Computing Spectral embedding")
	embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
										  eigen_solver="arpack")
	X_se = embedder.fit_transform(X)

	plot_3D_labeled_datapoints(X_se, y, y_label)

	#----------------------------------------------------------------------
	# t-SNE embedding of the digits dataset
	print("Computing t-SNE embedding")
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	X_tsne = tsne.fit_transform(X)

	plot_3D_labeled_datapoints(X_tsne, y, y_label)
	
def plot_3D_labeled_datapoints(X, y, y_label, savefig=None):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	legend = ["No spot", "Single spot", "Multi-spots"]
	colors = ['r', 'b', 'g']
	markers = ['^', 'o', 'x']
	for i in range(3):
		index = np.where(y_label == i)
		ax.scatter(X[index,0], X[index,1], y[index], c=colors[i], marker=markers[i], s=10, label=legend[i])
	plt.legend()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('Profile Score')
	
	if savefig is not None:
		plt.savefig(savefig)
		plt.close()
	else:
		plt.show()
		
def plot_3D_datapoints(X, y, savefig=None):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:,0], X[:,1], y, c='b', marker='o', s=10)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('Profile Score')
	
	if savefig is not None:
		plt.savefig(savefig)
		plt.close()
	else:
		plt.show()
