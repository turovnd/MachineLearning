import numpy as np
import matplotlib.patches
import pylab as pl


class Plot(object):
    def __init__(self):
        pass

    @staticmethod
    def smv(X1_train, X2_train, clf, dot, dotCl):
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
        pl.scatter(clf.sv[:, 0], clf.sv[:, 1], s=100, c="g")

        if dotCl == 1.0:
            pl.plot(dot[0], dot[1], "r*")
        else:
            pl.plot(dot[0], dot[1], "b*")

        X1, X2 = np.meshgrid(np.linspace(-1.1, 1.1, 50), np.linspace(-1.1, 1.1, 50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()

    @staticmethod
    def knn(X1_train, X2_train, distance, dot, dotCl):
        pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
        pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")

        dst = np.array([distance[i][2] for i in range(len(distance))])
        pl.scatter(dst[:, 0], dst[:, 1], s=100, c="g")

        if dotCl == 1.0:
            pl.plot(dot[0], dot[1], "r*")
        else:
            pl.plot(dot[0], dot[1], "b*")

        circle = matplotlib.patches.Circle(dot, float(distance[len(distance) - 1][0]), fill=False)
        pl.gca().add_patch(circle)

        pl.axis("tight")
        pl.show()
