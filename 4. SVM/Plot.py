import numpy as np
import pylab as pl


class Plot(object):
    def __init__(self):
        pass

    @staticmethod
    def smv(X1_train, X2_train, clf):
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()
