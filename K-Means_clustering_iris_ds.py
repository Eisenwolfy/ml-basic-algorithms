import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



class KMeans:
    def __init__(self, k: int = 3, max_iters: int = 300, tol: float = 1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iters_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        X = self._validate(X)

        if X.shape[0] < self.k:
            raise ValueError

        run_num_gen = np.random.default_rng(self.random_state)
        self.centroids_ = self._init_centroids(X, run_num_gen)

        for i in range(self.max_iters):
            labels = self._assign(X)

            new_centroids = self._update(X, labels)

            shift = np.linalg.norm(new_centroids - self.centroids_)

            self.centroids_ = new_centroids
            self.n_iters_ = i + 1

            if shift < self.tol:
                break

        self.labels_ = self._assign(X)
        self.inertia_ = self._compute_inertia(X, self.labels_)

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._validate(X)
        return self._assign(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = X.shape
        centroids = []

        first_idx = rng.integers(0, n_samples)
        centroids.append(X[first_idx].copy())

        for _ in range(1, self.k):
            cen_arr = np.array(centroids)
            d = X[:, np.newaxis, :] - cen_arr
            d_sq = (d**2).sum(axis=2).min(axis=1)

            prob = d_sq / d_sq.sum()

            new_idx = rng.choice(n_samples, p = prob)
            centroids.append(X[new_idx].copy())

        return np.array(centroids)


    def _assign(self, X: np.ndarray) -> np.ndarray:
        #assign each point to its nearest centroid
        d = X[:, np.newaxis, :] - self.centroids_
        d_sq = (d**2).sum(axis=2)

        return np.argmin(d_sq, axis=1)


    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        new_centroids = np.zeros((self.k, n_features)) #creating a placeholder

        for i in range(self.k):
            mask = (labels == i) #array of boolean values

            if mask.sum() > 0:
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                new_centroids[i] = self.centroids_[i]

        return new_centroids



    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        assigned_centroids = self.centroids_[labels]
        d = X - assigned_centroids
        return float((d**2).sum())

    @staticmethod
    def _validate(X) -> np.ndarray:
        #Here we are checking the data
        X = np.array(X, dtype = float)

        if X.ndim != 2:
            raise ValueError('We need 2 dimensions to cluster')

        if X.shape[0] == 0:
            raise ValueError("It's empty..")

        return X

    def _check_fitted(self):
        if self.centroids_ is None:
            raise RuntimeError

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KMeans(k={self.k}, max_iters={self.max_iters}, "
            f"tol={self.tol}, random_state={self.random_state})"
        )

    # ------------------------------------------------------------------
    # Silhouette metric
    # ------------------------------------------------------------------
    '''Now I want to check how good my clustering is'''
    def silhouette_score(self, X: np.ndarray) -> float:
        self._check_fitted()
        X = self._validate(X)

        labels = self.labels_
        n = X.shape[0]

        score = 0.0

        for i in range(n):
            same_cluster = labels == labels[i]

            if same_cluster.sum() > 1:
                a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1))
            else:
                a = 0

            b = np.inf

            for c in range(self.k):
                if c == labels[i]:
                    continue

                cluster_points = X[labels == c]
                if len(cluster_points) == 0:
                    continue

                dist = np.mean(np.linalg.norm(X[i] - cluster_points, axis=1))
                b = min(b, dist)

            score += (b - a) / max(a, b + 1e-9)

        return score / n




if __name__ == '__main__':
    iris = load_iris()
    X = iris.data

    model = KMeans(k=3, random_state=42)
    model.fit(X)
    labels = model.labels_
    print(f"{model.inertia_:.4f}")
    print(f"{model.silhouette_score(X)}")
'''
Now i want to visualize my clusters, but for this i need to shrink dataset 
because "iris" is 4D and i need 2D
The easiest way to do it - PCA`
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

centroids_2d = pca.transform(model.centroids_)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
plt.scatter(
    centroids_2d[:, 0],
    centroids_2d[:, 1],
    marker='x',
    s=200
)

plt.title("KMeans Clusters")
plt.show()