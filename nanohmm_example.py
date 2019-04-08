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
from __future__ import print_function

import nanohmm

A = [[0.5, 0.5], [0.0, 1.0]]
B = [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]
pi = [0.5, 0.5]

lambda_ = nanohmm.hmm_t(A, B, pi)


print("Forward:")
O = [0, 1, 0, 2]
f = nanohmm.forward_t(lambda_)
LL = nanohmm.forward(f, O)
print(LL)


print("Forward-Backward:")
fb = nanohmm.forwardbackward_t(lambda_)
LL = nanohmm.forwardbackward(fb, O)
print("beta:", fb.beta)
print(LL)


print("Viterbi:")
A = [[0.25, 0.75], [0.2, 0.8]]
B = [[0.65, 0.2, 0.15], [0.21, 0.29, 0.5]]
pi = [0.45, 0.55]

lambda_ = nanohmm.hmm_t(A, B, pi)
O = [0, 1, 2]
v = nanohmm.viterbi_t(lambda_)
LL, Q = nanohmm.viterbi(v, O)

print("Log-likelihood:", LL)
print("Probability:", 2**LL)
print("Q =", Q)


print("Baum--Welch:")
A = [[0.5, 0.5], [0.0, 1.0]]
B = [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]
pi = [0.5, 0.5]

lambda_ = nanohmm.hmm_t(A, B, pi)
bw = nanohmm.baumwelch_t(lambda_)

O = [0, 1, 0, 2]
LL, lambda_ = nanohmm.baumwelch(bw, O, 100)
print("LL =", LL)
print("Trained HMM:")
print("A = ", lambda_.A)
print("B = ", lambda_.B)
print("pi = ", lambda_.pi)
