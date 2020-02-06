#!/usr/bin/env python
"""
Relief family algorithm implementations.
=====

   Copyright 2017 Alfredo Mungo <alfredo.mungo@protonmail.ch>

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.


Algorithms
-----

This module provides implementation for the following algorithms:
    * Relief
    * ReliefF
    * RReliefF

Parallel execution is available through the `multiprocessing` module.


References
-----

The following research papers have been used during development:
    * Robnik-Sikonja, Marko & Kononenko, Igor. (2000). An adaptation of Relief
        for attribute estimation in regression. ICML '97: Proceedings of the
        Fourteenth International Conference on Machine Learning.
    * Robnik-Sikonja, Marko & Kononenko, Igor. (2003). Theoretical and Empirical
        Analysis of ReliefF and RReliefF. Machine Learning. 53. 23-69.
        10.1023/A:1025667309714.
"""


import os
import sys
from functools import reduce
from collections import deque
from multiprocessing import Pool
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


_PY2 = sys.version_info.major == 2
_STRING_TYPES = (str, unicode) if _PY2 else (str,)


class Relief(BaseEstimator, TransformerMixin):
    """Relief algorithm implementation.

    INSTANCE PROPERTIES
    ----
    w_: Weight vector
    n_iterations: Number of times to iterate
    n_features: Number of features to keep
    n_jobs: Number of concurrent jobs to use
    categorical: Iiterable of categorical feature indices
    random_state: RandomState instance used for the fitting step

    Categorical Features
    -----
    Categorical features are treated discretely even
    if their value is a floating point number. The difference
    funciton for categorical features returns only 1 or 0,
    respectively in the case in which the two values are different
    or equal.

    Weight Vector
    -----
    The Relief algorithm (and other variations) compute a weight
    vector ranking the importance of each feature. The weights can
    then be used to choose the most important features and discard
    the rest, reducing the feature space.
    """
    def __init__(self, **kwargs):
        """Initialise a new instance of this class.

        KEYWORD ARGUMENTS
        ----
        n_iterations: Number of times to iterate (defaults to 100)
        n_features: Number of features to keep (defaults to 1)
        n_jobs: Number of concurrent jobs to use (defaults to the number of available CPUs)
        categorical: Iiterable of categorical feature indices
        random_state: Seed to set before computing the weight vector or RandomState
            instance. If none is provided, a new RandomState instance is
            initialised.
        """
        kwargs = dict(kwargs)
        self.w_ = None

        def gen_random_state(rnd_state):
            """Generate random state instance"""
            if isinstance(rnd_state, np.random.RandomState):
                return rnd_state

            return np.random.RandomState(seed=rnd_state)

        for name, default_value, convf in (
                # Param name, default param value, param conversion function
                ('categorical', (), tuple),
                ('n_jobs', os.cpu_count(), int),
                ('n_iterations', 100, int),
                ('n_features', 1, int),
                ('random_state', None, gen_random_state)
        ):
            setattr(self, name, convf(kwargs.setdefault(name, default_value)))
            del kwargs[name]

        if self.n_jobs < 1:
            raise ValueError('n_jobs must be greater than 0')

        if kwargs:
            raise ValueError('Invalid arguments: %s' % ', '.join(kwargs))

    def fit(self, data, y):
        """Compute feature weights.

        ARGUMENTS
        ----
        data: The input data matrix
        y: The label vector
        """
        n, m = data.shape # Number of instances & features

        # Initialise state
        js = self.random_state.randint(n, size=self.n_iterations)

        # Compute weights
        if self.n_jobs > 1:
            results = deque()
            n_iterations = [int(np.floor(self.n_iterations / self.n_jobs))] * self.n_jobs
            n_iterations[-1] += int(np.floor(self.n_iterations % self.n_jobs))

            with Pool(processes=self.n_jobs) as pool:
                for n_iter, n_proc_iters in enumerate(n_iterations):
                    results.append(
                        pool.apply_async(
                            self._fit_iteration,
                            (data, y, n_iter * n_iterations[0], n_proc_iters, js)
                        )
                    )

                pool.close()
                pool.join()

            self.w_ = reduce(
                lambda a, res: a + res.get(),
                results,
                np.array([0.] * m)
            )
        else:
            self.w_ = self._fit_iteration(data, y, 0, self.n_iterations, js)

        self.w_ /= self.n_iterations

        return self

    def _fit_iteration(self, data, y, iter_offset, n_iters, js):
        w = np.array([0.] * data.shape[1])

        for i in range(iter_offset, n_iters + iter_offset):
            j = js[i]
            ri = data[j] # Random sample instance
            hit, miss = self._nn(data, y, j)

            w += np.array([
                self._diff(k, ri[k], miss[k])
                - self._diff(k, ri[k], hit[k])
                for k in range(data.shape[1])
            ])

        return w

    def _nn(self, data, y, j):
        """Return nearest instances from `data` (not) belonging to the `j`-th class.

        ARGUMENTS
        -----
        data: A numpy array
        y: The numpy array of boolean labels
        j: The index of the element to find the nearest neighbors of

        RETURN VALUE
        -----
        A tuple (h, m) of the nearest hits and misses from `data`.
        """
        ri = data[j]
        d = np.sum(
            np.array([
                self._diff(c, ri[c], data[:, c]) for c in range(len(ri))
            ]).T,
            axis=1
        )

        odata = data[d.argsort()]
        oy = y[d.argsort()]

        h = odata[oy == y[j]][0:1]
        m = odata[oy != y[j]][0]

        h = h[1] if h.shape[0] > 1 else h[0]

        return h, m

    def _diff(self, c, a1, a2):
        """Return the difference between the same attribute of two instances.

        ATTRIBUTES
        -----
        c: Feature index of the compared values
        a1: Element of the first instance to compare
        a2: Element of the second instance to compare

        RETURN VALUE
        -----
        A value in the range 0..1, where 0 means the values are the same and
        1 means they are maximally distant.
        """
        return (
            np.abs(a1 - a2) if c not in self.categorical
            else 1 - (a1 == a2)
        )

    def transform(self, data):
        """Transform the input data.

        This method uses the computed weight vector to produce a new
        dataset exhibiting only the `self.n_features` best-ranked features.

        ARGUMENTS
        -----
        data: The input data to transform

        RETURN VALUE
        -----
        A matrix with the same number of rows as `data` and at most
        `self.n_features` columns.
        """
        n_features = np.round(
            data.shape[1] * self.n_features
        ).astype(np.int16) if self.n_features < 1 else self.n_features
        feat_indices = np.flip(np.argsort(self.w_), 0)[0:n_features]

        return data[:, feat_indices]

    @property
    def support_(self):
        n_features = np.round(
            data.shape[1] * self.n_features
        ).astype(np.int16) if self.n_features < 1 else self.n_features
        feat_indices = np.flip(np.argsort(self.w_), 0)[0:n_features]
        
        mask = np.zeros(len(self.w_), dtype=np.bool)
        mask[feat_indices] = True

        return mask


class ReliefF(Relief):
    """ReliefF algorithm implementation.

    This is an improved version of the Relief algorithm, providing increased
    robustness and extended functionality by using k nearest neighbors per class
    and allowing for non-binary labelling.


    INSTANCE PROPERTIES
    -----
    k: The number of nearest neighbors to use per class
    approx_decimals: The number of decimals to approximate real values to
        when computing probabilities

    (See also `Relief`)
    """
    def __init__(self, **kwargs):
        """Initialise a new instance of this class.

        KEYWORD ARGUMENTS
        -----
        k: The number of nearest neighbors to use per class (defaults to 10)
        approx_decimals: The number of decimals to approximate real values to
            when computing probabilities (defaults to 4)
        ramp: True to use the ramp-based distance function (defaults to False)
        tdiff, teq: Only useful when `ramp=True`, these two parameters represent
            respectively the minimum distance threshold at which two values are
            treated as equal (distance will be 0) and the maximum distance
            threshold at which two values are treated as maximally different
            (distance will be 1).

        Ramp-based Distance Function
        -----
        This kind of distance function reduced the penalty introduced to
        continuous features. See the paper about ReliefF analysis for more
        details.

        (See also `Relief.__init__()`)
        """
        for name, default_value, convf in (
                # Param name, default param value, param conversion function
                ('k', 10, int),
                ('approx_decimals', 4, int),
                ('ramp', False, bool),
                ('tdiff', .1, float),
                ('teq', .01, float)
        ):
            setattr(self, name, convf(kwargs.setdefault(name, default_value)))
            del kwargs[name]

        if self.tdiff <= self.teq:
            raise ValueError('tdiff must be greater than teq')

        super(ReliefF, self).__init__(**kwargs)

    def fit(self, data, y):
        """Compute feature weights.

        ARGUMENTS
        ----
        data: The input data matrix
        y: The label vector
        """
        n, m = data.shape # Number of instances & features
        probs = self._class_frequencies(y)

        # Initialise state
        js = self.random_state.randint(n, size=self.n_iterations)

        # Compute weights
        if self.n_jobs > 1:
            results = deque()
            n_iterations = [int(np.floor(self.n_iterations / self.n_jobs))] * self.n_jobs
            n_iterations[-1] += int(np.floor(self.n_iterations % self.n_jobs))

            with Pool(processes=self.n_jobs) as pool:
                for n_iter, n_proc_iters in enumerate(n_iterations):
                    results.append(
                        pool.apply_async(
                            self._fit_iteration,
                            (data, y, n_iter * n_iterations[0], n_proc_iters, probs, js)
                        )
                    )

                pool.close()
                pool.join()

            self.w_ = reduce(
                lambda a, res: a + res.get(),
                results,
                np.array([.0] * m)
            )
        else:
            #self.w_ = np.array([.0] * m)
            self.w_ = self._fit_iteration(data, y, 0, self.n_iterations, probs, js)

        self.w_ /= self.n_iterations * self.k

        return self

    def _fit_iteration(self, data, y, iter_offset, n_iters, probs, js):
        m = data.shape[1]
        w = np.array([.0] * m)

        for i in range(iter_offset, iter_offset + n_iters):
            j = js[i]
            _knn = self._knn(data, y, j, probs.keys())
            hit = _knn[y[j]]

            del _knn[y[j]]
            miss = _knn

            for k in range(m):
                w[k] += (
                    np.array([
                        self._diff(j, m_idx, data, y)
                        for m_indices in miss.values()
                        for m_idx in m_indices
                    ]).sum()
                    - np.array([self._diff(j, h_idx, data, y) for h_idx in hit]).sum()
                )

        return w

    def _class_frequencies(self, y):
        """Return a dictionary containing the _frequency for each class."""
        return {c: self._frequency(c, y) for c in set(y)}

    def _knn(self, data, y, j, classes):
        """Return nearest instances from `data` belonging to each class.

        ARGUMENTS
        -----
        data: Normalised input data
        y: Label vector
        j: Instance to compute the k-NNs for
        classes: Set of classes

        RETURN VALUE
        -----
        A dictionary {c: _knn} containing at most `k` nearest neighbor indices
        for each class. `_knn` is an iterable of NNs for class `c`.
        """
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(j, r[0], data, y),
                    enumerate(data)
                )
            )
        ) # Distances from pivot_row
        sindices = dist.argsort() # Sorted indices
        oy = y[sindices]

        return {
            c: sindices[oy == c][slice(0, self.k) if c != y[j]\
            else slice(1, self.k+1)] for c in classes
        }

    def _frequency(self, value, vector):
        """Computes and returns the frequency ratio of `value` in `vector`."""
        if vector.shape[0] == 0:
            return 0

        if not isinstance(value, _STRING_TYPES):
            vector = np.around(vector.astype(np.float32), decimals=self.approx_decimals)

        return (vector == value).sum() / len(vector)

    def _diff(self, j1, j2, data, y):
        """Compute the difference between the `j1`-th and `j2`-th instances
            of `data`.

        ARGUMENTS
        -----
        j1, j2: Indices of instances to compute the difference for
        data: Input data matrix
        y: Label vector
        """
        return np.array([self._diff_value(c, j1, j2, data, y) for c in range(data.shape[1])]).sum()

    def _diff_ramp(self, a1, a2):
        """Perform the ramp-based difference on two values."""
        d = np.abs(a1 - a2)

        return (d - self.teq) / (self.tdiff - self.teq) if self.teq < d <= self.tdiff\
            else (0 if d <= self.teq else 1)

    def _diff_value(self, c, i1, i2, data, y):
        """Compute the difference between an attribute of two instances.

        ARGUMENTS
        -----
        c: Feature index
        i1, i2: Indices of the two instances in `data`
        data: Input data matrix
        y: Label array
        """
        v1 = data[i1, c]
        v2 = data[i2, c]
        na_v1 = self._isna(v1)
        na_v2 = self._isna(v2)
        categorical = c in self.categorical

        if not (na_v1 or na_v2):
            res = (
                (
                    self._diff_ramp(v1, v2) if self.ramp\
                    else np.abs(v1 - v2)
                ) if not categorical\
                else 1. - (v1 == v2)
            )
        else:
            if na_v1 != na_v2:
                res = self._diff_one_nan(
                    c,
                    i1 if na_v2 else i2,
                    i2 if na_v2 else i1,
                    data,
                    y
                )
            else:
                res = self._diff_both_nan(c, i1, i2, data, y)

        return res

    def _diff_one_nan(self, c, i_known, i_unknown, data, y):
        """Same as `_diff_value()` for one NaN value."""
        v_known = data[i_known, c]
        y_unknown = y[i_unknown]

        return 1. - self._frequency(v_known, data[y == y_unknown, c])

    def _diff_both_nan(self, c, i1, i2, data, y):
        """Same as `_diff_value()` for both NaN values."""
        y1 = y[i1]
        y2 = y[i2]

        return 1. - np.array([
            self._frequency(v, data[y == y1, c])
            * self._frequency(v, data[y == y2, c])
            for v in set(data[:, c])
            if not self._isna(v)
        ]).sum()

    @classmethod
    def _isna(cls, value):
        """
        Return a boolean value stating whether a value (scalar or vector) is NA.
        """
        if np.isscalar(value) or value is None:
            res = (
                (value is None or np.isnan(value)) if np.isreal(value) or np.iscomplex(value)\
                else False
            )
        else:
            res = np.frompyfunc(cls._isna, 1, 1)(value)

        return res


class RReliefF(ReliefF):
    """RReliefF algorithm implementation.

    The ReliefF algorithm is an extension of ReliefF allowing for
    ranking of continuously-labelled data.

    INSTANCE PROPERTIES
    -----
    sigma: Distance factor scaling hyper-parameter

    Sigma
    -----
    The sigma parameter allows for distances between instances to be deemed
    more or less important for computing weight vectors. The algorithm uses
    nearest-neighbor ranks instead of norm-based distances between neighbors
    in order to ensure that distances on different scales are treated equally.
    This strategy allows the distances to be computed in advance of the actual
    neigbors and are the same during the execution of the entire algorithm. To
    see the effect of sigma, see `RReliefF.distance_factors()`.

    (See also `ReliefF`)
    """
    def __init__(self, **kwargs):
        """Initialises a new instance of this class.

        ARGUMENTS
        -----
        sigma: Distance factor scaling hyper-parameter

        (See also `ReliefF.__init__()`)
        """

        # Set hyperparameters and prevent propagation to superclass
        kwargs = dict(kwargs)

        for name, default_value, convf in (
                # Param name, default param value, param conversion function
                ('sigma', .1, float),
        ):
            setattr(self, name, convf(kwargs.setdefault(name, default_value)))
            del kwargs[name]

        super(RReliefF, self).__init__(**kwargs)

    def fit(self, data, y):
        """Compute feature weights.

        ARGUMENTS
        ----
        data: The input data matrix
        y: The label vector
        """
        n, m = data.shape # Number of instances & features
        probs = self._class_frequencies(y)

        # Initialise state
        js = self.random_state.randint(n, size=self.n_iterations)
        ds = self.distance_factors()

        # Compute weights
        if self.n_jobs > 1:
            ndc = 0.
            nda = np.array([0.] * m)
            ndcda = np.array([0.] * m)
            n_iterations = [int(np.floor(self.n_iterations / self.n_jobs))] * self.n_jobs
            n_iterations[-1] += int(np.floor(self.n_iterations % self.n_jobs))
            results = deque()

            with Pool(processes=self.n_jobs) as pool:
                for n_iter, n_proc_iters in enumerate(n_iterations):
                    results.append(
                        pool.apply_async(
                            self._fit_iteration,
                            (data, y, n_iter * n_iterations[0], n_proc_iters, probs, js, ds)
                        )
                    )

                pool.close()
                pool.join()

            # Aggregate results
            for res in results:
                ndc_i, nda_i, ndcda_i = res.get()
                ndc += ndc_i
                nda += nda_i
                ndcda += ndcda_i
        else: # Single process mode - don't spawn anything
            ndc, nda, ndcda = self._fit_iteration(data, y, 0, self.n_iterations, probs, js, ds)

        self.w_ = ndcda / ndc - (nda - ndcda) / (self.n_iterations - ndc)

        return self

    def _fit_iteration(self, data, y, iter_offset, n_iters, probs, js, ds):
        """Run a single iteration of the algorithm.

        ARGUMENTS
        -----
        data: Input data matrix
        y: Label vector
        i: Iteration number
        probs: Probability vector
        js: Vector of randomly generated feature indices
        ds: Distance factors vector

        RETURN VALUE
        -----
        A tuple containing the ndc, nda, ndcda values.
        """
        m = data.shape[1]

        ndc = 0.
        nda = np.array([0.] * m)
        ndcda = np.array([0.] * m)

        for i in range(iter_offset, iter_offset + n_iters):
            j = js[i]
            _knn = self._knn(data, y, j) # Indices of the first `k` NN of `ri`

            for q in range(self.k):
                d = ds[q]
                diff_label = np.abs(y[j] - y[_knn[q]])
                ndc += diff_label * d

                for k in range(m):
                    nda_incr = self._diff_value(k, j, _knn[q], data, y) * d
                    nda[k] += nda_incr
                    ndcda[k] += diff_label * nda_incr

        return ndc, nda, ndcda

    def distance_factors(self):
        """Compute and return the distance factors vector"""
        d1 = np.exp(-((np.arange(self.k) + 1.) / self.sigma) ** 2)

        return d1 / d1.sum()

    def _knn(self, data, y, j):
        """Return the `self.k` nearest neighbors for the `j`-th instance in `data`.

        RETURN VALUE
        -----
        An iterable of at most `self.k` nearest neighbor indices in `data`.
        """
        dist = np.array(
            list(
                map(
                    lambda r: self._diff(j, r, data, y),
                    range(data.shape[0])
                )
            )
        ) # Distances from pivot_row

        return dist.argsort()[1:self.k+1] # Sorted indices
