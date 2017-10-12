import numpy as np


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
        if self.n_dims == 1:
            log_prob = c - (test_data - self.mu) ** 2.0 / (2.0 * self.sigma_sq)
        else:
            log_prob = c - np.sum((test_data - self.mu) ** 2.0 / (2.0 * self.sigma_sq), axis=1)

        return log_prob


class parzen:
    def __init__(self, n_dims, sigma):
        self.n_dims = n_dims
        self.sigma = sigma
        self.train_data = 0

    def train(self, train_data):
        self.train_data = train_data

    def compute_predictions(self, test_data):

        deno = (2.0*np.pi) ** (self.n_dims/2.0)
        deno1 = self.sigma ** self.n_dims
        c = 1 / float(deno * deno1)
        n = 1 / float(test_data.shape[0])
        log_prob = np.zeros(test_data.shape[0])

        for i, k in enumerate(test_data):
            acc = 0
            for x in self.train_data:
                norm_sq = np.linalg.norm((k-x)) ** 2
                sigma_sq = self.sigma ** 2
                p = (-0.5) * norm_sq / float(sigma_sq)
                acc += np.exp(p)
            if acc == 0:
                log_prob[i] = -np.inf
            else:
                log_prob[i] = np.log(n) + np.log(c) + np.log(acc)
        return log_prob
