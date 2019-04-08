# Copyright 2019 Vladimir Sukhoy and Alexander Stoytchev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math as m


def safelog2(x):
    """
    Computes the logarithm base 2 of the number x.

    Returns negative infinity when x is zero.
    """
    if x == 0:
        return -float('inf')
    else:
        return m.log(x, 2)


class hmm_t(object):
    """
    This class specifies an HMM that is described by its state transition
    probability matrix A, the observation probability matrix B, and the initial
    state probability vector pi.
    """
    def __init__(self, A, B, pi):
        """
        Initializes a new hmm_t object given the state transition probability
        matrix A, the observation probability matrix B, and the vector of
        initial state probabilities pi.
        """
        if len(A) != len(B) or len(A) != len(pi):
            raise ValueError("inconsistent number of states")
        if not B or any(len(_) != len(B[0]) for _ in B[1:]):
            raise ValueError("inconsistent alphabet size")
        self.A = A
        self.B = B
        self.pi = pi

    @property
    def N(self):
        """ Gets the value of N, i.e., the number of states in the hmm. """
        return len(self.A)

    @property
    def M(self):
        """ Gets the value of M, i.e., the size of the alphabet. """
        return len(self.B[0])


class forward_t(object):
    """ This class implements the Forward algorithm. """
    def __init__(self, lambda_, T=None):
        """
        Initializes a new `forward_t` instance using an `hmm_t` object and,
        optionally, the sequence length T.
        """
        self.lambda_ = lambda_
        if T:
            self.alpha = [[None]*T for _ in range(lambda_.N)]
            self.c = [None]*T
        else:
            self.alpha = self.c = None

    def __call__(self, O):
        """ Calls the Forward algorithm. """
        T = len(O)
        h = self.lambda_
        N = h.N
        M = h.M

        if not self.alpha or T > len(self.alpha[0]):  # resize if necessary
            self.alpha = [[None]*T for _ in range(N)]
        if not self.c or T > len(self.c):
            self.c = [None]*T

        alpha = self.alpha
        c = self.c

        logL = 0
        for t in range(T):
            assert O[t] in range(M)
            c[t] = 0
            for i in range(N):
                # Use pi instead of the recursive formula if t is zero.
                if 0 == t:
                    # Compute the alphas before renormalizing.
                    alpha[i][0] = h.pi[i] * h.B[i][O[0]]
                else:
                    # Use the recursive formula to compute alpha at time t in
                    # terms of alphas at time t-1.
                    alpha[i][t] = 0
                    for j in range(N):
                        alpha[i][t] += alpha[j][t-1] * h.A[j][i] * h.B[i][O[t]]
                c[t] += alpha[i][t]  # update the re-normalization coefficient
            if c[t] == 0 or float('inf') == 1.0/c[t]:
                c[t] = 1.0  # set c's to 1s to avoid NaNs
                # Set the log-likelihood to -inf to indicate that the sequence
                # is improbable. Otherwise, compute the renormalization
                # coefficient c[t] as usual.
                logL = -float('inf')
            else:
                c[t] = 1.0/c[t]
                for i in range(N):
                    alpha[i][t] *= c[t]  # renormalize the alphas
                logL -= safelog2(c[t])
        return logL



class forwardbackward_t(forward_t):
    """ This class implements the Forward-Backward algorithm. """
    def __init__(self, lambda_, T=None):
        """
        Initializes a new `forwardbackward_t` instance using an `hmm_t` object
        and, optionally, the sequence length T.
        """
        forward_t.__init__(self, lambda_, T)
        self.beta = [[None]*T for _ in range(lambda_.N)] if T else None


    def __call__(self, O):
        """ Calls the Forward-Backward algorithm. """
        T = len(O)
        h = self.lambda_
        N = h.N
        M = h.M

        logL = forward_t.__call__(self, O)

        alpha = self.alpha
        c = self.c

        if not self.beta or T > len(self.beta[0]):  # resize if necessary
            self.beta = [[None]*T for _ in range(N)]
        beta = self.beta

        for i in range(N):  # backward
            beta[i][T-1] = c[T-1]  # initialize renormalized betas
        for t in range(T-1, 0, -1):
            for i in range(N):
                beta[i][t-1] = 0
                for j in range(N):
                    # Update the beta at time t-1.
                    beta[i][t-1] += h.A[i][j] * h.B[j][O[t]] * beta[j][t]

                # Renormalize betas with the c's computed in the forward
                # iteration.
                beta[i][t-1] *= c[t-1]

        return logL


class viterbi_t(object):
    """ This class implements the Viterbi algorithm. """
    def __init__(self, lambda_, T=None):
        """
        Initializes a new `viterbi_t` instance using an `hmm_t` object and,
        optionally, the sequence length T.
        """
        self.lambda_ = lambda_
        self.delta_next = [None]*lambda_.N
        self.delta = [None]*lambda_.N
        self.psi = [[None]*T for _ in range(lambda_.N)] if T else None

    def __call__(self, O):
        """ Calls the Viterbi algorithm. """
        T = len(O)
        h = self.lambda_
        N = h.N
        M = h.M

        pi = h.pi
        A = h.A
        B = h.B

        delta = self.delta
        delta_next = self.delta_next

        if not self.psi or T > len(self.psi[0]):  # resize if necessary
            self.psi = [[None]*T for _ in range(N)]
        psi = self.psi

        for i in range(N):
            delta[i] = safelog2(pi[i]) + safelog2(B[i][O[0]])
            psi[i][0] = 0

        for t in range(1, T):
            for i in range(N):
                maxLL = -float('inf')
                argmax = 0

                for j in range(N):
                    LL = delta[j] + safelog2(A[j][i])
                    if LL > maxLL:
                        maxLL = LL
                        argmax = j

                delta_next[i] = maxLL + safelog2(B[i][O[t]])
                psi[i][t] = argmax

            delta, delta_next = delta_next, delta

        maxLL = -float('inf')
        for i in range(N):
            if delta[i] > maxLL:
                maxLL = delta[i]
                argmax = i

        Q = [None] * T
        for t in range(T-1, 0, -1):
            Q[t] = argmax
            argmax = psi[argmax][t]
        Q[0] = argmax

        return maxLL, Q


def normalize(x):
    """ Normalizes a vector so that it sums up to 1. """
    s = sum(x)
    n = len(x)
    return [1.0/n for _ in range(n)] if s == 0 else [_/s for _ in x]


class baumwelch_t(forwardbackward_t):
    """ This class implements the Baum--Welch algorithm. """
    def __init__(self, lambda_, T=None):
        """
        Initializes a new `baumwelch_t` instance using an `hmm_t` object and,
        optionally, the sequence length T.
        """
        forwardbackward_t.__init__(self, lambda_, T)
        self.sgamma = [None]*lambda_.N

    def __call__(self, O, num_iter):
        """ Invokes the Baum-Welch algorithm. """
        T = len(O)
        h = self.lambda_
        N = h.N
        M = h.M

        A = h.A
        B = h.B
        pi = h.pi

        a = [[None]*N for _ in range(N)]
        b = [[None]*M for _ in range(N)]

        sgamma = self.sgamma

        # This optimized version of the Baum-Welch algorithm does not store the
        # lists gamma and xi in memory. Their elements are computed on the fly
        # and used immediately to update the HMM model. A list of length N called
        # 'sgamma' is used to hold the cumulative sums of gammas for each state.
        for _ in range(num_iter):
            forwardbackward_t.__call__(self, O)
            alpha = self.alpha
            beta = self.beta
            c = self.c

            for i in range(N):
                sgamma[i] = 0
                for j in range(N):
                    a[i][j] = 0
                for j in range(M):
                    b[i][j] = 0

            for t in range(T):
                for i in range(N):
                    gamma = alpha[i][t] * beta[i][t] / c[t]  # compute gamma for state i at time t
                    if 0 == t:
                        h.pi[i] = gamma  # output the element of pi for state i during the first iteration
                        h.pi = normalize(h.pi)
                    if t == T-1:  # normalize and output the As during the last iteration
                        if sgamma[i] > 0:  # do not output the As for this state if the sum of gammas is zero
                            for j in range(N):
                                A[i][j] = a[i][j] / sgamma[i]
                        A[i] = normalize(A[i])
                    else:  # for every iteration except the last, add xi-s to the corresponding elements of a
                        for j in range(N):
                            a[i][j] += alpha[i][t] * A[i][j] * B[j][O[t+1]] * beta[j][t+1]
                    sgamma[i] += gamma
                    b[i][O[t]] += gamma

            for i in range(N):  # normalize and output the Bs after the last iteration
                if sgamma[i] > 0:  # do not output the Bs for this state if the sum of gammas is zero
                    for j in range(M):
                        B[i][j] = b[i][j] / sgamma[i]  # normalize by the sum of gammas up to time T
                B[i] = normalize(B[i])
        return forwardbackward_t.__call__(self, O), h


def forward(f, O):
    """
    This convenience function runs the Forward algorithm in a way that looks
    similar to the C version of the library.

    Parameters
    ----------
    f : forward_t
        Specifies the context for the Forward algorithm.
    O : sequence of integers between 0 and M-1
        Specifies the sequence of observations for the Forward algorithm.

    Returns
    -------
    log_likelihood : float
                     Log-likelihood (base 2) of the observation sequence.
    """
    return f(O)


def forwardbackward(fb, O):
    """
    This convenience function runs the Forward-Backward algorithm in a way that
    looks similar to the C version of the library.

    Parameters
    ----------
    fb : forwardbackward_t
         Specifies the context for the Forward-Backward algorithm.
    O  : sequence of integers between 0 and M-1
         Specifies the sequence of observations for the Forward-Backward
         algorithm.

    Returns
    -------
    log_likelihood : float
                     Log-likelihood (base 2) of the observation sequence.
    """
    return fb(O)


def viterbi(v, O):
    """
    This convenience function runs the Viterbi algorithm in a way that looks
    similar to the C version of the library.

    Parameters
    ----------
    v : viterbi_t
        Specifies the context for the Viterbi algorithm.
    O : sequence of integers between 0 and M-1
        Specifies the sequence of observations for the Viterbi algorithm.

    Returns
    -------
    log_likelihood : float
                     Log-likelihood (base 2) of the most likely state sequence.
    Q : list of integers between 0 and N-1
        The most likely state sequence generated by the Viterbi algorithm.
    """
    return v(O)


def baumwelch(bw, O, num_iter):
    """
    This convenience function runs the Baum--Welch algorithm in a way that looks
    similar to the C version of the library.

    Parameters
    ----------
    bw : baumwelch_t
         Specifies the context for the Baum--Welch algorithm.
    O : sequence of integers between 0 and M-1
        Specifies the sequence of observations for the Baum--Welch algorithm.

    Returns
    -------
    log_likelihood : float
                     Log-likelihood (base 2) of the sequence given the re-estimated HMM.
    lambda_ : hmm_t
              The re-estimated HMM.
    """
    return bw(O, num_iter)
