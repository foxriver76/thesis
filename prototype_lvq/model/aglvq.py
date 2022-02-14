# -*- coding: utf-8 -*-

# Orig. Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
# Adopted for data streams by Moritz Heusinger <moritz.heusinger@gmail.com>
# License: BSD 3 clause

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted
from itertools import product
from sklearn.utils.multiclass import unique_labels
import math


def _squared_euclidean(a, b=None):
    if b is None:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(a ** 2, 1) - 2 * a.dot(
            a.T)
    else:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(b ** 2, 1) - 2 * a.dot(
            b.T)
    return np.maximum(d, 0)


class GLVQ(ClassifierMixin, BaseEstimator):
    """
    Generalized Learning Vector Quantization
    Parameters
    ----------
    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.
    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.
    beta : int, optional (default=2)
        Used inside phi.
        1 / (1 + np.math.exp(-beta * x))
    C : array-like, shape = [2,3] ,optional
        Weights for wrong classification of form (y_real,y_pred,weight)
        Per default all weights are one, meaning you only need to specify
        the weights not equal one.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    decay_rate : float, Decay rate for Adadelta
    Attributes
    ----------
    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes
    classes_ : array-like, shape = [n_classes]
        Array containing labels.
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 beta=2, C=None, random_state=None, gradient_descent='SGD',
                 decay_rate=0.9, beta1=0.9, beta2=0.999, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.prototypes_per_class = prototypes_per_class
        self.initial_prototypes = initial_prototypes
        self.beta = beta
        self.epsilon = 1e-8
        self.c = C
        self.random_state = random_state
        self.initial_fit = True
        self.classes_ = []
        self.decay_rate = decay_rate
        allowed_gradient_descent = ['SGD', 'Adadelta', 'Adamax']
        if gradient_descent in allowed_gradient_descent:
            self.gradient_descent = gradient_descent
        else:
            raise ValueError('{} is not a valid parameter for '
                             'gradient_descent, please use one '
                             'of the following parameters:\n {}'
                             .format(gradient_descent, allowed_gradient_descent))

    def phi_prime(self, x):
        """
        Parameters
        ----------
        x : input value
        """
        return self.beta * np.math.exp(-self.beta * x) / (
                1 + np.math.exp(-self.beta * x)) ** 2

    def _validate_train_parms(self, train_set, train_lab, classes=None):
        if not isinstance(self.beta, int):
            raise ValueError("beta must a an integer")
        random_state = validation.check_random_state(self.random_state)
        train_set, train_lab = validation.check_X_y(train_set, train_lab)

        if (self.initial_fit):
            if (classes):
                self.classes_ = np.asarray(classes)
                self.protos_initialized = np.zeros(self.classes_.size)
            else:
                self.classes_ = unique_labels(train_lab)
                self.protos_initialized = np.zeros(self.classes_.size)

        nb_classes = len(self.classes_)
        nb_samples, nb_features = train_set.shape  # nb_samples unused

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            if self.prototypes_per_class < 0 or not isinstance(
                    self.prototypes_per_class, int):
                raise ValueError("prototypes_per_class must be a positive int")
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        else:
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    "values in prototypes_per_class must be positive")
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    "length of prototypes per class"
                    " does not fit the number of classes"
                    "classes=%d"
                    "length=%d" % (nb_classes, nb_ppc.size))
        # initialize prototypes
        if self.initial_prototypes is None:
            if self.initial_fit:
                self.w_ = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
                self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClassIdx in range(len(self.classes_)):
                actClass = self.classes_[actClassIdx]
                nb_prot = nb_ppc[actClassIdx]  # nb_ppc: prototypes per class
                if (self.protos_initialized[actClassIdx] == 0 and actClass in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == actClass, :], 0)
                    self.w_[pos:pos + nb_prot] = mean + (
                            random_state.rand(nb_prot, nb_features) * 2 - 1)
                    if math.isnan(self.w_[pos, 0]):
                        print('Prototype is NaN: ', actClass)
                        self.protos_initialized[actClassIdx] = 0
                    else:
                        self.protos_initialized[actClassIdx] = 1

                    self.c_w_[pos:pos + nb_prot] = actClass
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        if self.initial_fit:
            self.squared_mean_gradient = np.zeros_like(self.w_)
            self.squared_mean_step = np.zeros_like(self.w_)
            self.past_squared_gradients_v = np.zeros_like(self.w_)
            self.past_gradients_m = np.zeros_like(self.w_)
            self.initial_fit = False

        ret = train_set, train_lab, random_state

        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))
        if self.c is not None:
            self.c = validation.check_array(self.c)
            if self.c.shape != (2, 3):
                raise ValueError("C must be shape (2,3)")
            for k1, k2, v in self.c:
                self.c_[tuple(zip(*product(self._map_to_int(k1), self._map_to_int(k2))))] = float(v)

        return ret

    def _map_to_int(self, item):
        return np.where(self.c_w_ == item)[0]

    def _optimize(self, x, y, random_state):
        training_data = x
        label_equals_prototype = y[np.newaxis].T == self.c_w_

        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = self.w_.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong
        mu = np.vectorize(self.phi_prime)(mu)

        g = np.zeros(prototypes.shape)
        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong

            dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
            dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
            g[i] = dcd.dot(training_data[idxw]) - dwd.dot(
                training_data[idxc]) + (dwd.sum(0) -
                                        dcd.sum(0)) * prototypes[i]
        g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        g = g * (1 + 0.0001 * (random_state.rand(*g.shape) - 0.5))

        if self.gradient_descent == 'SGD':
            self.w_ -= g.ravel().reshape(self.w_.shape)
        elif self.gradient_descent == 'Adadelta':
            # Accumulate gradient
            self.squared_mean_gradient = self.decay_rate * self.squared_mean_gradient + \
                                         (1 - self.decay_rate) * g ** 2

            # Compute update/step
            step = - ((self.squared_mean_step + self.epsilon) / \
                      (self.squared_mean_gradient + self.epsilon)) ** 0.5 * g

            # Accumulate updates
            self.squared_mean_step = self.decay_rate * self.squared_mean_step + \
                                     (1 - self.decay_rate) * step ** 2
            # step = np.array([abs(s)*1 if y==self.c_w_[idx] else abs(s)*-1 for idx,s in enumerate(step)])
            self.w_ += step.ravel().reshape(self.w_.shape)
        else:
            # Adamax
            self.past_gradients_m = self.beta1 * self.past_gradients_m + \
                                    (1 - self.beta1) * g

            self.past_squared_gradients_v = np.maximum(self.beta2 * self.past_squared_gradients_v, np.linalg.norm(g))

            # Bias correction
            m_hat = self.past_gradients_m / (1 - self.beta1)

            # Perform update
            self.w_ -= (self.learning_rate / (self.past_squared_gradients_v + self.epsilon)) * m_hat

    def _compute_distance(self, x, w=None):
        if w is None:
            w = self.w_
        return cdist(x, w, 'euclidean')

    def predict(self, x):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        #        check_is_fitted(self, ['w_', 'c_w_'])
        #        x = validation.check_array(x)
        #        if x.shape[1] != self.w_.shape[1]:
        #            raise ValueError("X has wrong number of features\n"
        #                             "found=%d\n"
        #                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)
        return (self.c_w_[dist.argmin(1)])

    def decision_function(self, x):
        """Predict confidence scores for samples.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.check_array(x)
        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)

        foo = lambda c: dist[:, self.c_w_ != self.classes_[c]].min(1) - dist[:, self.c_w_ == self.classes_[c]].min(1)
        res = np.vectorize(foo, signature='()->(n)')(self.classes_).T

        if self.classes_.size <= 2:
            return res[:, 1]
        else:
            return res

    def get_info(self):
        return 'GLVQ'

    def fit(self, X, y, classes=None):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)
        Returns
        --------
        self
        """
        self.initial_fit = True

        X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        self._optimize(X, y, random_state)

        return self

    def partial_fit(self, X, y, classes=None):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)
        Returns
        --------
        self
        """
        if set(unique_labels(y)).issubset(set(self.classes_)) or self.initial_fit == True:
            X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'.format(y))

        self._optimize(X, y, random_state)
        return self

    def reset(self):
        raise NotImplementedError()

    def predict_proba(self):
        raise NotImplementedError()

    def get_prototypes(self):
        return self.w_