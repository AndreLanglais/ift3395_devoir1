# -*- coding: utf-8 -*-
import numpy as np
import pylab
import utilitaires
import graphic_plotter
import estimators

class classif_bayes:

    def __init__(self, modeles_diag, priors):
        self.modeles_diag = modeles_diag
        self.priors = priors
        if len(self.modeles_diag) != len(self.priors):
            print 'Le nombre de modeles MV doit etre egale au nombre de priors!'

        self.n_classes = len(self.modeles_diag)

    # Retourne une matrice de taille nb. ex. de test x nombre de classes contenant les log
    # probabilités de chaque exemple de test sous chaque modèle MV.
    def compute_predictions(self, test_data, eval_by_group=False):
        log_pred = np.empty((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # ici il va falloir utiliser modeles_mv[i] et priors pour remplir
            # chaque colonne de log_pred (c'est plus efficace de faire tout une
            # colonne a la fois)

            log_pred[:, i] = self.modeles_diag[i].compute_predictions(test_data) + np.log(self.priors[i])

        return log_pred


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

iris_train = np.concatenate([iris_train1, iris_train2, iris_train3])
iris_test = np.concatenate([iris_test1, iris_test2, iris_test3])

#Graphiques estimateur 1d
graphic_plotter.plot_1d(iris_train1)

#Graphiques estimateur 2d
graphic_plotter.plot_2d(iris_train1)

#Bayes Diagonal 2d

def estimate_gauss(train_cols):
    model_classe1_diag = estimators.gauss_diag(len(train_cols))
    model_classe2_diag = estimators.gauss_diag(len(train_cols))
    model_classe3_diag = estimators.gauss_diag(len(train_cols))

    model_classe1_diag.train(iris_train1[:, train_cols])
    model_classe2_diag.train(iris_train2[:, train_cols])
    model_classe3_diag.train(iris_train3[:, train_cols])

    models_diag = [model_classe1_diag, model_classe2_diag, model_classe3_diag]
    priors = [0.3333, 0.3333, 0.3333]

    classifieur = classif_bayes(models_diag, priors)

    # calcul des probs
    log_prob_train = classifieur.compute_predictions(iris_train[:, train_cols])
    log_prob_test = classifieur.compute_predictions(iris_test[:, train_cols])

    #
    classesPred_train = log_prob_train.argmax(1) + 1
    classesPred_test = log_prob_test.argmax(1) + 1

    if len(train_cols) == 2:
        utilitaires.gridplot(classifieur,
                            iris_train[:, train_cols + [-1]],
                            iris_test[:, train_cols + [-1]],
                            n_points=50)

    print "Bayes Diagonal " + str(len(train_cols)) + "D"
    print "Taux d'erreur (entrainement) %.2f%%" % ((1 - (classesPred_train == iris_train[:, -1]).mean()) * 100.0)
    print "Taux d'erreur (test) %.2f%%" % ((1 - (classesPred_test == iris_test[:, -1]).mean()) * 100.0)


#Bayes 2D
train_cols = [0, 1]
estimate_gauss(train_cols)

#Bayes 4D

train_cols = [0, 1, 2, 3]
estimate_gauss(train_cols)

# Bayes Parzen 2d

def estimate_parzen(train_cols, sigma):

    model_classe1_parzen = estimators.parzen(len(train_cols), sigma)
    model_classe2_parzen = estimators.parzen(len(train_cols), sigma)
    model_classe3_parzen = estimators.parzen(len(train_cols), sigma)

    model_classe1_parzen.train(iris_train1[:, train_cols])
    model_classe2_parzen.train(iris_train2[:, train_cols])
    model_classe3_parzen.train(iris_train3[:, train_cols])

    # liste modeles et priors
    models_parzen = [model_classe1_parzen, model_classe2_parzen, model_classe3_parzen]
    priors = [0.3333, 0.3333, 0.3333]

    # classifieur
    classifieur = classif_bayes(models_parzen, priors)

    # calcul des probs
    log_prob_train = classifieur.compute_predictions(iris_train[:, train_cols])
    log_prob_test = classifieur.compute_predictions(iris_test[:, train_cols])

    classesPred_train = log_prob_train.argmax(1) + 1
    classesPred_test = log_prob_test.argmax(1) + 1

    utilitaires.gridplot(classifieur,
                         iris_train[:, train_cols + [-1]],
                         iris_test[:, train_cols + [-1]],
                         n_points=50)

    print "Parzen " + str(sigma)
    print "Taux d'erreur (entrainement) %.2f%%" % ((1 - (classesPred_train == iris_train[:, -1]).mean()) * 100.0)
    print "Taux d'erreur (test) %.2f%%" % ((1 - (classesPred_test == iris_test[:, -1]).mean()) * 100.0)

# Parzen sigma petit
train_cols = [0,1]
sigma_petit = 0.01
sigma_grand = 0.075
sigma_app = 0.02

estimate_parzen(train_cols, sigma_petit)
estimate_parzen(train_cols, sigma_grand)
estimate_parzen(train_cols, sigma_app)

# Courbes apprentissages

def learning_curve(train_cols, sigmas):
    error_train = np.empty(sigmas.shape[0])
    error_val = np.empty(sigmas.shape[0])
    for i, k in enumerate(sigmas):
        model_classe1_parzen = estimators.parzen(len(train_cols), k)
        model_classe2_parzen = estimators.parzen(len(train_cols), k)
        model_classe3_parzen = estimators.parzen(len(train_cols), k)

        model_classe1_parzen.train(iris_train1[:, train_cols])
        model_classe2_parzen.train(iris_train2[:, train_cols])
        model_classe3_parzen.train(iris_train3[:, train_cols])

        # liste modeles et priors
        models_diag = [model_classe1_parzen, model_classe2_parzen, model_classe3_parzen]
        priors = [0.3333, 0.3333, 0.3333]

        # classifieur
        classifieur = classif_bayes(models_diag, priors)

        # calcul des probs
        log_prob_train = classifieur.compute_predictions(iris_train[:, train_cols])
        log_prob_test = classifieur.compute_predictions(iris_test[:, train_cols])

        #
        classesPred_train = iris_train.shape[0] - np.sum((log_prob_train.argmax(1) + 1) == iris_train[:, -1])
        classesPred_test = iris_test.shape[0] - np.sum((log_prob_test.argmax(1) + 1) == iris_test[:, -1])
        error_train[i] = classesPred_train
        error_val[i] = classesPred_test

        success_train = log_prob_train.argmax(1) + 1
        success_test = log_prob_test.argmax(1) + 1

        print k
        print "Taux d'erreur (entrainement) %.2f%%" % ((1 - (success_train == iris_train[:, -1]).mean()) * 100.0)
        print "Taux d'erreur (test) %.2f%%" % ((1 - (success_test == iris_test[:, -1]).mean()) * 100.0)

    return error_train, error_val

# Courbes 2D
sigmas = np.linspace(0.01, 0.2, 100)
train_cols = [0, 1]

curve_data = learning_curve(train_cols, sigmas)

pylab.plot(sigmas, curve_data[0])
pylab.plot(sigmas, curve_data[1])
pylab.show()

# Courbes 4D
sigmas = np.linspace(0.1, 1, 100)
train_cols = [0, 1, 2, 3]

curve_data = learning_curve(train_cols, sigmas)

pylab.plot(sigmas, curve_data[0])
pylab.plot(sigmas, curve_data[1])
pylab.show()
