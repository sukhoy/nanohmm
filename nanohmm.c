// Copyright 2019 Vladimir Sukhoy and Alexander Stoytchev
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "nanohmm.h"
#include <assert.h>
#include <math.h>


double forward(forward_t *f, const unsigned int *O, const unsigned int T) {
  const hmm_t *h = f->lambda;
  const unsigned int N = h->N;
  const unsigned int M = h->M;

  double **alpha = f->alpha;
  double *c = f->c;

  assert(h != 0 && T > 0 && O != 0 && alpha != 0 && c != 0);
  assert(M > 0 && N > 0 && h->A != 0 && h->B != 0 && h->pi != 0);
  unsigned int t, i, j;
  double logL = 0.0;
  for (t=0; t<T; ++t) {   // forward
    assert(O[t] < M);
    c[t] = 0.0;
    for (i=0; i<N; ++i) {
      if (0 == t) // use pi instead of the recursive formula if t is zero
        alpha[i][0] = h->pi[i] * h->B[i][O[0]]; // compute the alpha before renormalizing
      else {  // use the recursive formula to compute alpha at time t in terms of alphas at time t-1
        alpha[i][t] = 0.0;
        for (j=0; j<N; ++j)       // compute alpha values before renormalizing
          alpha[i][t] += (alpha[j][t-1]) * (h->A[j][i]) * (h->B[i][O[t]]);
      }
      c[t] += alpha[i][t]; // update the re-normalization coefficient
    }
    if (c[t] == 0.0 || HUGE_VAL == 1.0/c[t]) { // The seq. is improbable if c[t] is very close to 0.
      c[t] = 1.0;       // set c's to 1's to avoid NaN's
      logL = -HUGE_VAL; // set log-likelihood= -infinity to indicate that the sequence is improbable
    } else {            // otherwise, compute the re-normalization coefficient c[t] as usual
      c[t] = 1.0/c[t];
      for (i=0; i<N; ++i)
        alpha[i][t] *= c[t]; // renormalize the alphas
      logL -= log2(c[t]);     // update the return value
    }
  }

  return logL;
}


double forwardbackward(forwardbackward_t *fb,
                       const unsigned int *O,
                       const unsigned int T) {
  const hmm_t *h = fb->lambda;
  const unsigned int N = h->N;
  const unsigned int M = h->M;

  double **alpha = fb->alpha;
  double *c = fb->c;
  double **beta = fb->beta;

  assert(h != 0 && T > 0 && O != 0 && alpha != 0 && beta != 0 && c != 0);
  assert(M > 0 && N > 0 && h->A != 0 && h->B != 0 && h->pi != 0);
  unsigned int t, i, j;
  double logL = forward((forward_t*)fb, O, T);
  for (i=0; i<N; ++i)     // backward
    beta[i][T-1] = c[T-1];   // initialize renormalized betas
  for (t=T-1; t>0; t--) {
    for (i=0; i<N; ++i) { // compute beta at time t-1 using betas at time t
      beta[i][t-1] = 0.0;
      for (j=0; j<N; ++j)
        beta[i][t-1] += (h->A[i][j]) * (h->B[j][O[t]]) * beta[j][t]; // update the beta at time t-1
      beta[i][t-1] *= c[t-1]; // renormalize betas with the c-s computed in the forward iteration
    }
  }
  return logL;
}


double viterbi(viterbi_t *v,
               const unsigned int *O,
               const unsigned int T,
               unsigned int *Q) {
  unsigned int N = v->lambda->N;
  unsigned int M = v->lambda->M;
  const double *pi = v->lambda->pi;
  double** const A = v->lambda->A;
  double** const B = v->lambda->B;

  double *delta = v->delta;
  double *delta_next = v->delta_next;
  unsigned int **psi = v->psi;

  unsigned int i, j, t;
  for (i=0; i<N; ++i) {
    delta[i] = log2(pi[i]) + log2(B[i][O[0]]);
    psi[i][0] = 0;
  }

  double maxLL;
  unsigned int argmax;
  for (t=1; t<T; ++t) {
    for (i=0; i<N; ++i) {
      maxLL = -HUGE_VAL;
      argmax=0;

      for (j=0; j<N; ++j) {
        double LL = delta[j] + log2(A[j][i]);
        if (LL > maxLL) {
          maxLL = LL;
          argmax = j;
        }
      }

      delta_next[i] = maxLL + log2(B[i][O[t]]);
      psi[i][t] = argmax;
    }

    double *tmp = delta;
    delta = delta_next;
    delta_next = tmp;
  }

  maxLL = -HUGE_VAL;
  for (i=0; i<N; ++i) {
    if (delta[i] > maxLL) {
      maxLL = delta[i];
      argmax = i;
    }
  }

  for (t=T-1; t>=1; --t) {
    Q[t] = argmax;
    argmax = psi[argmax][t];
  }
  Q[0] = argmax;

  return maxLL;
}



static void normalize(double *x, unsigned int n) {
  assert(n > 0);
  double sum = 0.0;
  unsigned int ix;
  for (ix=0; ix<n; ++ix) {
    assert(x[ix] >= 0.0);
    sum += x[ix];
  }
  for (ix=0; ix<n; ++ix)                                // normalize the array of nonnegative 
    x[ix] = (sum == 0.0) ? (1.0/n):(x[ix]/sum);         // values x so that it sums up to 1
}


double baumwelch(baumwelch_t *bw,
                 const unsigned int *O,
                 const unsigned int T,
                 unsigned int num_iter) {
  const hmm_t *h = bw->lambda;
  const unsigned int N = h->N;
  const unsigned int M = h->M;

  double **alpha = bw->alpha;
  double *c = bw->c;
  double **beta = bw->beta;
  double *sgamma = bw->sgamma;

  // This optimized version of the Baum-Welch algorithm does not store the
  // arrays gamma and xi in memory. Their elements are computed on the fly
  // and used immediately to update the HMM model. An array of length N called
  // 'sgamma' is used to hold the cumulative sums of gammas for each state.
  assert(h != 0 && T > 0 && O != 0 && num_iter > 0);
  unsigned int i, t, j;
  double a[N][N];       // re-estimated state transition matrix before it is normalized
  double b[N][M];       // re-estimated observation probability matrix before it is normalized
  double gamma;               // current value of gamma for each state
  while (num_iter--) {
    // It is assumed that fb renormalizes at every step so that the xis are already normalized
    // and gammas can be computed as alpha[i][t] * beta[i][t] / c[t].
    forwardbackward((forwardbackward_t*)bw, O, T);
    for (i=0; i<N; ++i) {     // zero the accumulated values
      sgamma[i] = 0.0;
      for (j=0; j<N; ++j)
        a[i][j] = 0.0;
      for (j=0; j<M; ++j)
        b[i][j] = 0.0;
    }
    for (t=0; t<T; ++t) {
      for (i=0; i<N; ++i) {
        gamma = alpha[i][t] * beta[i][t] / c[t];     // compute gamma for state i at time t
        if (0 == t) {
          h->pi[i] = gamma;       // output the element of pi for state i during the first iteration
          normalize(h->pi, N);
        }
        if (t == T-1) {        // normalize and output the As during the last iteration
          if (sgamma[i] > 0.0) // do not output the As for this state if the sum of gammas is zero 
            for (j=0; j<N; ++j)   // normalize by the sum of gammas for all times up to now
              h->A[i][j] = (a[i][j] / sgamma[i]);
          normalize(h->A[i], N);
        }
        else {  // for every iteration except the last, add xi-s to the corresponding elements of a
          for (j=0; j<N; ++j)
            a[i][j] += alpha[i][t] * (h->A[i][j]) * h->B[j][O[t+1]] * beta[j][t+1];  // add xi to a
        }
        sgamma[i] += gamma;     // update the sum of gammas used for normalization
        b[i][O[t]] += gamma;    // add gamma to the corresponding element of b
      }
    }
    for (i=0; i<N; ++i) {    // normalize and output the Bs after the last iteration
      if (sgamma[i] > 0.0)      // do not output the Bs for this state if the sum of gammas is zero 
        for (j=0; j<M; ++j)
          h->B[i][j] = (b[i][j] / sgamma[i]);   // normalize by the sum of gammas up to time T
      normalize(h->B[i], M);
    }
  }
  // return the log-likelihood of the sequence, calcutated with the re-estimated A, B, and pi
  return forwardbackward((forwardbackward_t*)bw, O, T);
}


inline hmm_t* hmm_init_block(void *block, unsigned int N, unsigned int M) {
  hmm_t *rv = block;
  char *p = block;

  unsigned int ix;

  rv->M = M;
  rv->N = N;
  p += sizeof(hmm_t);

  // Initialize the N x N matrix A.
  rv->A = (double**)p;
  p += sizeof(double*)*N;
  for (ix=0; ix<N; ++ix) {
    rv->A[ix] = (double*)p;
    p += sizeof(double)*N;
  }

  // Initialize the N x M matrix B.
  rv->B = (double**)p;
  p += sizeof(double*)*N;
  for (ix=0; ix<N; ++ix) {
    rv->B[ix] = (double*)p;
    p += sizeof(double)*M;
  }

  // Initialize the vector pi.
  rv->pi = (double*)p;

  return rv;
}


forward_t* forward_init_block(void *block,
                              const hmm_t *lambda,
                              const unsigned int T,
                              unsigned int hsize) {
  if (hsize == 0)
    hsize = sizeof(forward_t);

  forward_t *rv = block;
  char *p = block + hsize;

  unsigned int ix;

  rv->lambda = lambda;

  // Initialize the N x T matrix alpha.
  rv->alpha = (double**)p;
  p += sizeof(double*)*lambda->N;
  for (ix=0; ix<lambda->N; ++ix) {
    rv->alpha[ix] = (double*)p;
    p += sizeof(double)*T;
  }

  // Initialize the vector c.
  rv->c = (double*)p;

  return rv;
}


forwardbackward_t* forwardbackward_init_block(void *block,
                                              const hmm_t *lambda,
                                              const unsigned int T,
                                              unsigned int hsize) {
  if (hsize == 0)
    hsize = sizeof(forwardbackward_t);

  forwardbackward_t *rv = block;
  char *p;
  unsigned int ix;

  forward_init_block(block, lambda, T, hsize);

  p = (char*)rv->c + sizeof(double)*T;  // continue from c

  rv->beta = (double**)p;
  p += sizeof(double*)*lambda->N;
  for (ix=0; ix<lambda->N; ++ix) {
    rv->beta[ix] = (double*)p;
    p += sizeof(double)*T;
  }

  return rv;
}


viterbi_t* viterbi_init_block(void *block, const hmm_t *lambda,
                              const unsigned int T,
                              unsigned int hsize) {
  if (hsize == 0)
    hsize = sizeof(viterbi_t);

  viterbi_t *rv = block;
  char *p = block + hsize;

  unsigned int ix;

  rv->lambda = lambda;

  rv->delta = (double*)p;
  p += sizeof(double)*lambda->N;
  rv->delta_next = (double*)p;
  p += sizeof(double)*lambda->N;

  rv->psi = (unsigned int**)p;
  p += sizeof(unsigned int*)*lambda->N;
  for (ix=0; ix<lambda->N; ++ix) {
    rv->psi[ix] = (unsigned int*)p;
    p += sizeof(unsigned int)*T;
  }

  return rv;
}


baumwelch_t* baumwelch_init_block(void *block,
                                  const hmm_t *lambda,
                                  const unsigned int T,
                                  unsigned int hsize) {
  if (hsize == 0)
    hsize = sizeof(baumwelch_t);

  baumwelch_t *rv = block;

  forwardbackward_init_block(block, lambda, T, hsize);

  char *p = (char*)rv->beta[lambda->N-1] + sizeof(double)*T;  // continue from beta
  rv->sgamma = (double*)p;

  return rv;
}
