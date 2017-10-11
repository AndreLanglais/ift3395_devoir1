# -*- coding: utf-8 -*-
import numpy as np
import pylab
import estimators


def plot_1d(train_data):
    # Graphique estimateur 1d
    train_cols = [0]
    sample = np.linspace(min(train_data[:, train_cols]), max(train_data[:, train_cols]), 500)
    # Affichage de l'estimateur gaussien diagonal
    model = estimators.gauss_diag(1)

    model.train(train_data[:, train_cols])

    gaussien_prob = np.exp(model.compute_predictions(sample))
    pylab.plot(sample, gaussien_prob, color="red", label="Gaussian")

    # Affichage de l'estimateur de Parzen avec noyau gaussien isotropique
    low_sigma = 0.01
    high_sigma = 0.075
    good_sigma = 0.030

    parzen1 = estimators.parzen(1, low_sigma)
    parzen2 = estimators.parzen(1, high_sigma)
    parzen3 = estimators.parzen(1, good_sigma)

    parzen1.train(train_data[:, train_cols])
    parzen2.train(train_data[:, train_cols])
    parzen3.train(train_data[:, train_cols])

    parzen1_prob = np.exp(parzen1.compute_predictions(sample))
    parzen2_prob = np.exp(parzen2.compute_predictions(sample))
    parzen3_prob = np.exp(parzen3.compute_predictions(sample))

    pylab.plot(sample, parzen1_prob, color="blue", label="Parzen (sigma = " + str(low_sigma) + ")")
    pylab.plot(sample, parzen2_prob, color="orange", label="Parzen (sigma = " + str(high_sigma) + ")")
    pylab.plot(sample, parzen3_prob, color="green", label="Parzen (sigma = " + str(good_sigma) + ") appropriate parameter")
    pylab.legend(loc="upper left")
    # on affiche les points sur l'axe des x
    pylab.plot(train_data[:, 0], len(train_data) * [0], "o")

    pylab.show()


def plot_2d(train_data):
    train_cols = [0, 1]
    train_col_x = 0
    train_col_y = 1

    sample_x = pylab.linspace(min(train_data[:, train_col_x]), max(train_data[:, train_col_x]), 50)
    sample_y = pylab.linspace(min(train_data[:, train_col_y]), max(train_data[:, train_col_y]), 50)

    gauss_2d = estimators.gauss_diag(2)
    gauss_2d.train(train_data[:, train_cols])

    pylab.plot(train_data[:, train_col_x], train_data[:, train_col_y], "o", label="Gaussian")
    pylab.contour(sample_x, sample_y, create_contour_z(sample_x, sample_y, gauss_2d))
    pylab.legend()
    pylab.show()

    low_sigma = 0.01
    high_sigma = 0.075
    good_sigma = 0.02

    parzen1_2d = estimators.parzen(2, low_sigma)
    parzen3_2d = estimators.parzen(2, high_sigma)
    parzen2_2d = estimators.parzen(2, good_sigma)

    parzen1_2d.train(train_data[:, train_cols])
    parzen2_2d.train(train_data[:, train_cols])
    parzen3_2d.train(train_data[:, train_cols])

    pylab.plot(train_data[:, train_col_x], train_data[:, train_col_y], "o", label="Parzen 2d (sigma = " + str(low_sigma) + ")")
    pylab.contour(sample_x, sample_y, create_contour_z(sample_x, sample_y, parzen1_2d))
    pylab.legend()
    pylab.show()

    pylab.plot(train_data[:, train_col_x], train_data[:, train_col_y], "o", label="Parzen 2d (sigma = " + str(high_sigma) + ")")
    pylab.contour(sample_x, sample_y, create_contour_z(sample_x, sample_y, parzen3_2d))
    pylab.legend()
    pylab.show()

    pylab.plot(train_data[:, train_col_x], train_data[:, train_col_y], "o", label="Parzen 2d (sigma = " + str(good_sigma) + ") appropriate parameter")
    pylab.contour(sample_x, sample_y, create_contour_z(sample_x, sample_y, parzen2_2d))
    pylab.legend()
    pylab.show()


def create_contour_z(sample_x, sample_y, estimator):
    prob = np.ndarray((sample_x.shape[0], sample_y.shape[0]))
    for i, k in enumerate(sample_x):
        plot_data = np.transpose([np.tile(k, len(sample_y)), np.repeat(sample_y, 1)])
        prob[i] = np.exp(estimator.compute_predictions(plot_data))

    return prob


