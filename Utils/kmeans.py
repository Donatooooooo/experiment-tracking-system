from sklearn.cluster import KMeans
from kneed import KneeLocator

class kMeans:
    def clustering(self, X):
        k = self.computeK(X, max_k=12)
        kmeans = KMeans(n_clusters=k, n_init=5, init='random')
        kmeans.fit(X)
        return kmeans
    
    def computeK(self, X, max_k):
        distortions = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init=5, init='random')
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
            
        optimal_k = KneeLocator(range(1, max_k + 1), distortions, curve="convex", direction="decreasing")
        return optimal_k.knee