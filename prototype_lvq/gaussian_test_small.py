#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:11:56 2018

@author: moritz
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from arslvq import RSLVQ
from aglvq import GLVQ
from sklearn.metrics import accuracy_score

X, y_true = make_blobs(n_samples=3000, centers=4,
                       cluster_std=1.0, random_state=None)
X = X[:, ::-1] # flip axes for better plotting

clf = RSLVQ(prototypes_per_class=4, gradient_descent='SGD', sigma=1.0)
labels_new = clf.partial_fit(X=X, y=y_true).predict(X)


clf = RSLVQ(prototypes_per_class=4, gradient_descent='Adadelta', sigma=1.0, decay_rate=0.9)
labels_ada = clf.partial_fit(X=X, y=y_true).predict(X)


clf = RSLVQ(prototypes_per_class=4, gradient_descent='Adamax', sigma=1.0)
labels_adamax = clf.partial_fit(X=X, y=y_true).predict(X)

acc_new = accuracy_score(y_true, labels_new)
acc_ada = accuracy_score(y_true, labels_ada)
acc_adamax = accuracy_score(y_true, labels_adamax)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,11))

ax[0].scatter(X[:, 0], X[:, 1], c=labels_new, s=40, cmap='viridis')
ax[0].set_title('SGD')
ax[0].text(0.5, -0.1, 'Accuracy: {}'.format(round(acc_new, 4)), size=12, ha="center",
         transform=ax[0].transAxes)

ax[1].scatter(X[:, 0], X[:, 1], c=labels_ada, s=40, cmap='viridis')
ax[1].set_title('Adadelta')
ax[1].text(0.5, -0.1, 'Accuracy: {}'.format(round(acc_ada, 4)), size=12, ha="center",
         transform=ax[1].transAxes)

ax[2].scatter(X[:, 0], X[:, 1], c=labels_adamax, s=40, cmap='viridis')
ax[2].set_title('AdaMax')
ax[2].text(0.5, -0.1, 'Accuracy: {}'.format(round(acc_adamax, 4)), size=12, ha="center",
         transform=ax[2].transAxes)


plt.show()

# now the glvq

clf = GLVQ(prototypes_per_class=4, gradient_descent='SGD')
labels_new = clf.partial_fit(X=X, y=y_true).predict(X)

clf = GLVQ(prototypes_per_class=4, gradient_descent='Adadelta', decay_rate=0.9)
labels_ada = clf.partial_fit(X=X, y=y_true).predict(X)

clf = GLVQ(prototypes_per_class=4, gradient_descent='Adamax')
labels_adamax = clf.partial_fit(X=X, y=y_true).predict(X)

acc_new = accuracy_score(y_true, labels_new)
acc_ada = accuracy_score(y_true, labels_ada)
acc_adamax = accuracy_score(y_true, labels_adamax)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,11))

ax[0].scatter(X[:, 0], X[:, 1], c=labels_new, s=40, cmap='viridis')
ax[0].set_title('SGD')
ax[0].text(0.5, -0.1, 'Accuracy: {}'.format(round(acc_new, 4)), size=12, ha="center",
         transform=ax[0].transAxes)

ax[1].scatter(X[:, 0], X[:, 1], c=labels_ada, s=40, cmap='viridis')
ax[1].set_title('Adadelta')
ax[1].text(0.5, -0.1, 'Accuracy: {}'.format(round(acc_ada, 4)), size=12, ha="center",
         transform=ax[1].transAxes)

ax[2].scatter(X[:, 0], X[:, 1], c=labels_adamax, s=40, cmap='viridis')
ax[2].set_title('AdaMax')
ax[2].text(0.5, -0.1, 'Accuracy: {}'.format(round(acc_adamax, 4)), size=12, ha="center",
         transform=ax[2].transAxes)

plt.show()