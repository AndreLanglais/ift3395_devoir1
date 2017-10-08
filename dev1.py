# -*- coding: utf-8 -*-
import numpy as np
import pylab

class gauss_diag:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.mu = np.zeros((1, n_dims))
        self.sigma_sq = np.ones(n_dims)
        self.cov = np.ones((n_dims, n_dims))

    # Pour un ensemble d'entrainement, la fonction devrait calculer l'estimateur par MV de la moyenne et de la matrice de covariance
    def train(self, train_data):
        self.mu = np.mean(train_data, axis=0)
        self.sigma_sq = np.sum((train_data - self.mu) ** 2.0, axis=0) / train_data.shape[0]

    # Retourne un vecteur de taille nb. ex. de test contenant les log
    # probabilites de chaque exemple de test sous le modele.
    def compute_predictions(self, test_data):
        # on prend le produit du vecteur representant la diagonale (np.prod(self.sigma)
        c = -self.n_dims * np.log(2 * np.pi) / 2.0 - np.log(np.prod(self.sigma_sq)) / 2.0
        # on somme sur l'axe 1 apres avoir divise par sigma puisque celui ci aussi est
        # de dimension d
        log_prob = c - np.sum((test_data - self.mu) ** 2.0 / (2.0 * self.sigma_sq), axis=1)
        return log_prob

class parzen:
    def __init__(self, n_dims, sigma):
        self.n_dims = n_dims
        self.sigma = sigma
        self.train_data = 0

    def train(self, train_data):
        self.train_data = train_data

    def compute_predictions(self):
        pass


# On decoupe les donnees en train/test
iris=np.loadtxt('iris.txt')
np.random.seed(123)
indices1 = np.arange(0, 50)
indices2 = np.arange(50, 100)
indices3 = np.arange(100, 150)

np.random.shuffle(indices1)
np.random.shuffle(indices2)
np.random.shuffle(indices3)

iris_train1 = iris[indices1[:35]]
iris_test1 = iris[indices1[35:]]
iris_train2 = iris[indices2[:35]]
iris_test2 = iris[indices2[35:]]
iris_train3 = iris[indices3[:35]]
iris_test3 = iris[indices3[35:]]


model = gauss_diag(4)
model.train(iris_train1)

# on affiche les points sur l'axe des x
pylab.plot(iris_train1[:, 0], len(iris_train1) * [0], "o")


# on crée une distribution normal avec les paramètres qu'on a calculé et on pige des valeurs
mu = model.mu[0]
sigma = np.sqrt(model.sigma_sq[0])
s = np.random.normal(mu, sigma, 1000)
s.sort()

# pour chaque valeur de la distribution, on calcule la probabilité de celle-ci avec la fonction de densité calculé
pylab.plot(s, 1/(sigma * np.sqrt(2 * np.pi)) *
           np.exp(- (s - mu)**2 / (2 * sigma**2)))

pylab.show()
