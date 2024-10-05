import matplotlib.pyplot as  plot
import seaborn as sea
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
iris=load_iris()
data=iris.data
scaler=StandardScaler()
scaled=scaler.fit_transform(data)
Pca=PCA(n_components=2)
pca_data=Pca.fit_transform(scaled)
k_means=KMeans(n_clusters=3,random_state=42)
cluster=k_means.fit_predict(pca_data)
ev1=silhouette_score(pca_data,cluster)
ev2=davies_bouldin_score(pca_data,cluster)
print(ev1)
print(ev2)
sea.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster, palette="deep")
plot.title('Build an Unsupervised Learning Model on a Dataset(Clustering Model)')
plot.show()

#first we are adding the iris datasetand in the second step, we are cleaning rhe dataset
#the scaler will standardize the data and the fit() will find the mean and normalize the data and
#converts it into the transformed matrix. the PCA will reduce the dimensionality into 2X2 because the
#n_components are given as 2. it will convert.
#here we are using the k_means clustering model inwhich the distance nearer to it will add into it.
#here n=3 so 3 clusters willbe there.
#in the cluster variable the transformed k_means for pca willbe stored
#next the silhoutte score is presented. it is the measures how similar
# a point is to its own cluster compared to other clusters.
# Higher scores indicate better-defined clusters.
#next the davies_bouldinscore is calculated, it will evaluates clustering by measuring the
# average similarity ratio of each cluster with the cluster that is most similar to it.
# Lower scores indicate better clustering.
#hue=cluster will color accordingt o their cluster labels

#Brief conclusion
#if we look into the iris dataset we coulad find that the setosa have lesser sepal and petal dimensions
#iris versicolour is the intermediate and the virginica has the longest dimensions.
#the 0 represents the setosa, 1 represents the versicolor and the 1 represents the virginica
