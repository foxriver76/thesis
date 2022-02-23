#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:35:11 2018

@author: moritz
"""

import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted

# TODO: add sigma for every prototype (TODO from https://github.com/MrNuggelz/sklearn-lvq)

class RSLVQ(ClassifierMixin, BaseEstimator):
    """Robust Soft Learning Vector Quantization
    Parameters
    ----------
    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.
    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.
    sigma : float, optional (default=1.0)
        Variance for the distribution.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    gradient_descent : string, Gradient Descent describes the used technique
        to perform the gradient descent. Possible values: 'SGD' (default),
        'RMSprop', 'Adadelta'.
    Attributes
    ----------
    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features
    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes
    classes_ : array-like, shape = [n_classes]
        Array containing labels.
    initial_fit : boolean, indicator for initial fitting. Set to false after
        first call of fit/partial fit.
    """

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 sigma=1.0, gradient_descent='SGD', random_state=None,
                 decay_rate=0.9, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.sigma = sigma
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epsilon = 1e-8
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.initial_fit = True
        self.classes_ = []
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        allowed_gradient_descent = ['SGD', 'RMSprop', 'Adadelta', 'Adamax']
        if gradient_descent in allowed_gradient_descent:
            self.gradient_descent = gradient_descent
        else:
            raise ValueError('{} is not a valid parameter for '
                             'gradient_descent, please use one '
                             'of the following parameters:\n {}'
                             .format(gradient_descent, allowed_gradient_descent))
            
    def get_prototypes(self):
        """Returns the prototypes"""
        return self.w_

    def _optimize(self, x, y, random_state):            
        if (self.gradient_descent=='SGD'):
            """Implementation of Stochastical Gradient Descent"""
            n_data, n_dim = x.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = x[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])
                    c = 1 / self.sigma
                    if self.c_w_[j] == c_xi:
                        # Attract prototype to data point
                        self.w_[j] += c * (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                     self._p(j, xi, prototypes=self.w_)) * d
                    else:
                        # Distance prototype from data point
                        self.w_[j] -= c * self._p(j, xi, prototypes=self.w_) * d            
            
        elif (self.gradient_descent=='Adadelta'):
            """Implementation of Adadelta"""
            n_data, n_dim = x.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = x[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])
                    
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                     self._p(j, xi, prototypes=self.w_)) * d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_) * d
                        
                     # Accumulate gradient
                    self.squared_mean_gradient[j] = self.decay_rate * self.squared_mean_gradient[j] + \
                                (1 - self.decay_rate) * gradient ** 2
                    
                    # Compute update/step
                    step = - ((self.squared_mean_step[j] + self.epsilon) / \
                              (self.squared_mean_gradient[j] + self.epsilon)) ** 0.5 * gradient
                              
                    # Accumulate updates
                    self.squared_mean_step[j] = self.decay_rate * self.squared_mean_step[j] + \
                    (1 - self.decay_rate) * step ** 2
                    
                    # Attract/Distract prototype to/from data point
                    self.w_[j] -= step
        
        elif (self.gradient_descent=='RMSprop'):
            """Implementation of RMSprop"""
            n_data, n_dim = x.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = x[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = (xi - prototypes[j])
                    
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                     self._p(j, xi, prototypes=self.w_)) * d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_) * d
                        
                    # Accumulate gradient
                    self.squared_mean_gradient[j] = 0.9 * self.squared_mean_gradient[j] + \
                            0.1 * gradient ** 2
                    
                    # Update Prototype
                    self.w_[j] += (self.learning_rate / ((self.squared_mean_gradient[j] + \
                           self.epsilon) ** 0.5)) * gradient
        elif (self.gradient_descent=='Adamax'):
            """Implementation of AdaMax"""
            n_data, n_dim = x.shape
            nb_prototypes = self.c_w_.size
            prototypes = self.w_.reshape(nb_prototypes, n_dim)

            for i in range(n_data):
                xi = x[i]
                c_xi = y[i]
                for j in range(prototypes.shape[0]):
                    d = xi - prototypes[j]
        
                    if self.c_w_[j] == c_xi:
                        gradient = (self._p(j, xi, prototypes=self.w_, y=c_xi) -
                                     self._p(j, xi, prototypes=self.w_)) * d
                    else:
                        gradient = - self._p(j, xi, prototypes=self.w_) * d
                     
                    # Accumulate gradient
                    self.past_gradients_m[j] = self.beta1 * self.past_gradients_m[j] + \
                         (1 - self.beta1) * gradient
            
                    self.past_squared_gradients_v[j] = np.maximum(self.beta2 * self.past_squared_gradients_v[j], np.linalg.norm(gradient))
                        
                    # Bias correction
                    m_hat = self.past_gradients_m[j] / (1 - self.beta1)
            
                    # Perform update
                    self.w_[j] += (self.learning_rate / (self.past_squared_gradients_v[j] + self.epsilon)) * m_hat
     
    def _costf(self, x, w, **kwargs):
        d = (x - w)[np.newaxis].T 
        d = d.T.dot(d) 
        return -d / (2 * self.sigma)

    def _p(self, j, e, y=None, prototypes=None, **kwargs):
        if prototypes is None:
            prototypes = self.w_
        if y is None:
            fs = [self._costf(e, w, **kwargs) for w in prototypes]
        else:
            fs = [self._costf(e, prototypes[i], **kwargs) for i in
                  range(prototypes.shape[0]) if
                  self.c_w_[i] == y]

        fs_max = np.max(fs)
        s = sum([np.math.exp(f - fs_max) for f in fs])
        o = np.math.exp(
            self._costf(e, prototypes[j], **kwargs) - fs_max) / s
        return o

    def predict(self, x):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples)
            Returns predicted values.
        """
#        check_is_fitted(self, ['w_', 'c_w_'])
#        x = validation.check_array(x)
#        if x.shape[1] != self.w_.shape[1]:
#            raise ValueError("X has wrong number of features\n"
#                             "found=%d\n"
#                             "expected=%d" % (self.w_.shape[1], x.shape[1]))

        def foo(e):
            fun = np.vectorize(lambda w: self._costf(e, w),
                               signature='(n)->()')
            pred = fun(self.w_).argmax()
            return self.c_w_[pred]

        return np.vectorize(foo, signature='(n)->()')(x)

    def posterior(self, y, x):
        """
        calculate the posterior for x:
         p(y|x)
        Parameters
        ----------
        
        y: class
            label
        x: array-like, shape = [n_features]
            sample
        Returns
        -------
        posterior
        :return: posterior
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.column_or_1d(x)
        if y not in self.classes_:
            raise ValueError('y must be one of the labels\n'
                             'y=%s\n'
                             'labels=%s' % (y, self.classes_))
        s1 = sum([self._costf(x, self.w_[i]) for i in
                  range(self.w_.shape[0]) if
                  self.c_w_[i] == y])
        s2 = sum([self._costf(x, w) for w in self.w_])
        return s1 / s2
    
    def get_info(self):
        return 'RSLVQ'
    
    def predict_proba(self, X):
        """ predict_proba
        
        Predicts the probability of each sample belonging to each one of the 
        known target_values.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_values) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
        
        """
        return 'Not implemented'
    
    def reset(self):
        self.__init__()
        
    def _validate_train_parms(self, train_set, train_lab, classes=None):
        random_state = validation.check_random_state(self.random_state)
        train_set, train_lab = validation.check_X_y(train_set, train_lab)

        if(self.initial_fit):
            if (classes is not None):
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
            # nb_ppc = number of protos per class
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
            for actClass in range(len(self.classes_)):

                nb_prot = nb_ppc[actClass] # nb_ppc:  # prototypes per class
                if(self.protos_initialized[actClass] == 0 and self.classes_[actClass] in unique_labels(train_lab)):
                    mean = np.mean(
                        train_set[train_lab == self.classes_[actClass], :], 0)
                    self.w_[pos:pos + nb_prot] = mean + (
                            random_state.rand(nb_prot, nb_features) * 2 - 1)
                    if math.isnan(self.w_[pos, 0]):
                        self.protos_initialized[actClass] = 0
                    else:
                        self.protos_initialized[actClass] = 1
    
                    self.c_w_[pos:pos + nb_prot] = self.classes_[actClass]
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
            # Next two lines are Init for Adadelta/RMSprop
            self.squared_mean_gradient = np.zeros_like(self.w_)
            self.squared_mean_step = np.zeros_like(self.w_)
            self.past_squared_gradients_v = np.zeros_like(self.w_)
            self.past_gradients_m = np.zeros_like(self.w_)
            self.initial_fit = False

        return train_set, train_lab, random_state

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
        if unique_labels(y) in self.classes_ or self.initial_fit == True:
            X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'.format(y))
            
        self._optimize(X, y, random_state)
        return self
    
    def partial_fit(self, X, y, classes=None): # added
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
        if y.all() in self.classes_ or self.initial_fit == True:
            X, y, random_state = self._validate_train_parms(X, y, classes=classes)
        else:
            raise ValueError('Class {} was not learned - please declare all \
                             classes in first call of fit/partial_fit'.format(y))
            
        self._optimize(X, y, random_state)
        return self
    