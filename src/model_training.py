import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def find_best_k(X, y, max_k=30):
    """Find best K using 5-fold cross-validation."""
    k_range = range(1, max_k + 1)
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=5)
        k_scores.append(scores.mean())

    best_k = k_range[np.argmax(k_scores)]
    return best_k, k_range, k_scores


def train_knn(X_train, y_train, k):
    """Train final KNN model."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model
