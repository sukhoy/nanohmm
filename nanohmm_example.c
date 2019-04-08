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

// An example of how to use the nanohmm library.
#include "nanohmm.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// print a message to the standard error stream
static int emsg(const char *fmt, ...) {
  fprintf(stderr, "[nanohmm example] ");
  va_list args;
  va_start(args, fmt);
  int rv = vfprintf(stderr, fmt, args);
  va_end(args);
  fprintf(stderr, "\n");
  return rv;
}

// normalize the array of nonnegative values x so that it sums up to 1
static void normalize(double *x, unsigned int n) {
  assert(n > 0);
  double sum = 0.0;
  unsigned int ix;
  for (ix=0; ix<n; ++ix) {
    assert(x[ix] >= 0.0);
    sum += x[ix];
  }
  for (ix=0; ix<n; ++ix)
    x[ix] = (sum == 0.0) ? (1.0/n):(x[ix]/sum);
}


// check the invariants
static void check_hmm_invariants(const hmm_t *h) {
  const unsigned int N = h->N;
  const unsigned int M = h->M;
  const double INVTOL = 0.0000001;

  unsigned int i,j;
  double sum = 0;
  for (i=0; i<N; ++i)
    sum += h->pi[i];
  if (fabs(sum - 1.0) > INVTOL) {
    emsg("Initial state probabilities (pi) do not sum up to 1.0. \
They sum up to %f. Terminating the program.", sum);
    exit(-1);
  }
  
  for (i=0; i<N; ++i) {
    sum = 0;
    for (j=0; j<N; ++j)
      sum += h->A[i][j];
    if (fabs(sum - 1.0) > INVTOL) {
      emsg("Transition probabilities for state %d (0-based) do not sum up to 1.0. \
They sum up to %f. Terminating the program.", i, sum);
      exit(-1);
    }
    sum = 0;
    for (j=0; j<M; ++j)
      sum += h->B[i][j];
    if (fabs(sum - 1.0) > INVTOL) {
      emsg("Observation probabilities for state %d (0-based) do not sum up to 1.0. \
They sum up to %f. Terminating the program.", i, sum);
      exit(-1);
    }
  }
}


// Train an HMM using the Baum-Welch algorithm and random restarts.
// It is assumed that the memory for the HMM is already fully allocated.
static double train(hmm_t *h, const unsigned int T, const unsigned int *O,
      unsigned int num_restarts, unsigned int num_iters) {
  const unsigned int N = h->N;
  const unsigned int M = h->M;

  assert(NULL != h && 0 < N && 0 < M && NULL != h->A && NULL != h->B && NULL != h->pi);
  double bestA[N][N];
  double bestB[N][M];
  double bestpi[N];
  double bestLL = -HUGE_VAL;
  unsigned int ix, jx;

  char buffer[baumwelch_block_size(N, M, T)];
  baumwelch_t *bw = baumwelch_init_block(buffer, h, T, 0);

  while (num_restarts--) {
    for (ix=0; ix<N; ++ix) {
      h->pi[ix] = (double)rand() / RAND_MAX;
      for (jx=0; jx<N; ++jx)
        h->A[ix][jx] = (double)rand() / RAND_MAX;
      normalize(h->A[ix], N);
      for (jx=0; jx<M; ++jx)
        h->B[ix][jx] = (double)rand() / RAND_MAX;
      normalize(h->B[ix], M);
    }
    normalize(h->pi, N);
    const double LL = baumwelch(bw, O, T, num_iters);
    if (LL > bestLL) {
      bestLL = LL;
      for (ix=0; ix<N; ++ix) {
        bestpi[ix] = h->pi[ix];
        for (jx=0; jx<N; ++jx)
          bestA[ix][jx] = h->A[ix][jx];
        for (jx=0; jx<M; ++jx)
          bestB[ix][jx] = h->B[ix][jx];
      }
    }
  }
  for (ix=0; ix<N; ++ix) {
    h->pi[ix] = bestpi[ix];
    for (jx=0; jx<N; ++jx)
      h->A[ix][jx] = bestA[ix][jx];
    for (jx=0; jx<M; ++jx)
      h->B[ix][jx] = bestB[ix][jx];
  }

  // validate the HMM parameters.
  unsigned int i, j;
  for (i=0; i<N; ++i) {
    double sum = 0;
    for (j=0; j<N; ++j)
      sum += h->A[i][j];

    if (sum < 0.000001) {
      emsg("Training produced a transition probability matrix (A) that contains a row of zeros. Patching it.");
      assert(sum >= 0.0);
      // Sometimes Baum-Welch may produce a matrix A with a zero row.
      // If this happens, then simply set one of the entries to 1.0 and the rest
      // to 0 so that the invariant is maintained.
      h->A[i][0] = 1.0;
      for (j=1; j<N; ++j)
        h->A[i][j] = 0.0;
    }
    check_hmm_invariants(h);
  }
  return bestLL;
}

// print the parameters (A, B, and pi) of the HMM
static void hmm_print(const hmm_t *h) {
  unsigned int ix, jx;
  for (ix=0; ix<h->N; ++ix) {
    for (jx=0; jx<h->N; ++jx)
      printf("A[%d][%d] = %.4f\t", ix, jx, h->A[ix][jx]);
    printf("\n");
  }
  printf("\n");
  for (ix=0; ix<h->N; ++ix) {
    for (jx=0; jx<h->M; ++jx)
      printf("B[%d][%d] = %.4f\t", ix, jx, h->B[ix][jx]);
    printf("\n");
  }
  printf("\n");
  for (ix=0; ix<h->N; ++ix)
    printf("pi[%d]   = %.4f\t", ix, h->pi[ix]);
  printf("\n");
}


// compute the log-likelihood using forward-backward
static double compute_LL(const hmm_t *h, const unsigned int T, const unsigned int *O) {
  char f_block[forward_block_size(h->N, T)];  // allocate on stack
  forward_t *f = forward_init_block(f_block, h, T, 0);
  return forward(f, O, T);
}

// shows how to train an HMM from a fixed starting point without random restarts
static void bw_example() {
  printf("Baum-Welch example:\n");

  const unsigned int N = 2;
  const unsigned int M = 3;

  char hmm_block[hmm_block_size(N, M)];
  hmm_t *lambda = hmm_init_block(hmm_block, N, M);

  double **A = lambda->A;
  double **B = lambda->B;
  double *pi = lambda->pi;

  A[0][0] = 0.5;  A[0][1] = 0.5;
  A[1][0] = 0.0;  A[1][1] = 1.0;
  B[0][0] = 0.5;  B[0][1] = 0.5;  B[0][2] = 0.0;
  B[1][0] = 0.5;  B[1][1] = 0.0;  B[1][2] = 0.5;
  pi[0]   = 0.5;  pi[1]   = 0.5;

  const unsigned int O[] = {0, 1, 0, 2};
  const unsigned int T = sizeof(O) / sizeof(O[0]);

  printf("Running Baum-Welch from a fixed starting point.\n");

  char bw_buffer[baumwelch_block_size(N, M, T)];
  baumwelch_t *bw = baumwelch_init_block(bw_buffer, lambda, T, 0);

  const double LL = baumwelch(bw, O, T, 100);
  printf("Log-likelihood (base 2) of the sequence [");
  unsigned int ix;
  for (ix=0; ix<T; ++ix)
    printf("%d%s", O[ix], ix == T-1 ? "":" ");
  printf("] in the trained HMM is %g.\n", LL);
  printf("The parameters after training are:\n");
  hmm_print(lambda);
}

// shows training with random restarts and recognition using the trained HMMs
static void recognition_example() {
  printf("Recognition example:\n");
  const unsigned int example_O[][5] = {{0, 1, 2, 0, 1},            // the four training sequences
                                       {2, 1, 0, 0, 2},
                                       {2, 2, 2, 2, 2},
                                       {1, 1, 1, 2, 2}};

  const unsigned int O[][5] = {{0, 2, 1, 0, 1}, {1, 1, 2, 2, 2}};  // the two testing sequences
  const unsigned int T = 5;
  const unsigned int ntrain = sizeof(example_O)/sizeof(example_O[0]);
  const unsigned int ntest  = sizeof(O)/sizeof(O[0]);
  const unsigned int M = 3, N = 2;
  unsigned int train_ix, ix ,jx;
  printf("Training HMMs...\n");
  hmm_t* h[ntrain];
  for (train_ix=0; train_ix<ntrain; ++train_ix) {
    h[train_ix] = malloc(hmm_block_size(N, M));
    hmm_init_block(h[train_ix], N, M);

    train(h[train_ix], T, example_O[train_ix], 10, 50);  // 10 restarts, 50 iterations
    printf("Trained HMM %d on sequence [", train_ix + 1);
    for (jx=0; jx<T; ++jx)
      printf("%d%s", example_O[train_ix][jx], jx == T-1 ? "":" ");
    printf("].\n");
  }
  printf("Recognizing sequences...\n");
  unsigned int test_ix;
  for (test_ix=0; test_ix<ntest; ++test_ix) {
    unsigned int best_ix;
    double best_LL = -HUGE_VAL;
    for (ix=0; ix<ntrain; ++ix) {
      const double LL = compute_LL(h[ix], T, O[test_ix]);
      if (LL > best_LL) {
        best_LL = LL;
        best_ix = ix;
      }
    }
    printf("The HMM with the highest log-likelihood (%g) for the sequence [", best_LL);
    for (ix=0; ix<T; ++ix)
      printf("%d%s", O[test_ix][ix], ix == T-1 ? "":" ");
    printf("] is HMM number %d.\n", best_ix + 1);
  }

  for (train_ix=0; train_ix<ntrain; ++train_ix)
    free(h[train_ix]);
}


static void viterbi_example() {
  printf("Viterbi example:\n");
  char hmm_block[hmm_block_size(2, 3)];
  hmm_t *h = hmm_init_block(hmm_block, 2, 3);

  h->A[0][0] = 0.25; h->A[0][1] = 0.75;
  h->A[1][0] = 0.2; h->A[1][1] = 0.8;

  h->B[0][0] = 0.65; h->B[0][1] = 0.2; h->B[0][2] = 0.15;
  h->B[1][0] = 0.21; h->B[1][1] = 0.29; h->B[1][2] = 0.5;

  h->pi[0] = 0.45; h->pi[1] = 0.55;

  check_hmm_invariants(h);

  const unsigned int O[] = {0, 1, 2};
  const unsigned int T = sizeof(O) / sizeof(O[0]);

  char viterbi_block[viterbi_block_size(2, T)];
  viterbi_t *v = viterbi_init_block(viterbi_block, h, T, 0);

  unsigned int Q[T];

  const double LL = viterbi(v, O, T, Q);

  printf("The most probable state sequence is: [");
  unsigned int ix;
  for (ix=0; ix<T; ++ix)
    printf("%d%s", Q[ix], ix == T-1 ? "":" ");
  printf("].\n");
  printf("Its probability is: %g.\n", pow(2, LL));
  printf("\n");
  printf("\n");
}


int main(int argc, const char* argv[]) {
  const unsigned int N = 2;
  const unsigned int M = 3;
  char hmm_block[hmm_block_size(N, M)];
  hmm_t *lambda = hmm_init_block(hmm_block, N, M);
  double **A = lambda->A;
  double **B = lambda->B;
  double *pi = lambda->pi;
  // State transition probability matrix.
  A[0][0] = 0.5;  A[0][1] = 0.5;
  A[1][0] = 0.0;  A[1][1] = 1.0;
  // Observation probability matrix.
  B[0][0] = 0.5;  B[0][1] = 0.5;  B[0][2] = 0.0;
  B[1][0] = 0.5;  B[1][1] = 0.0;  B[1][2] = 0.5;
  // Initial state distribution vector.
  pi[0]   = 0.5;  pi[1]   = 0.5;
  const unsigned int O[] = {0, 1, 0, 2};  // observation sequence
  const unsigned int T = sizeof(O) / sizeof(O[0]);  // sequence length

  char f_block[forward_block_size(N, T)];  // allocate on the stack
  forward_t *f = forward_init_block(f_block, lambda, T, 0);  // initialize
  double LL = forward(f, O, T);
  printf("(forward) LL=%g\n", LL);

  char fb_block[forwardbackward_block_size(N, T)];  // allocate on stack
  forwardbackward_t *fb = forwardbackward_init_block(fb_block, lambda, T, 0);
  LL = forwardbackward(fb, O, T);

  printf("Forward-Backward\n");
  printf("Normalized betas:\n");
  unsigned int i, j;
  for (i=0; i<N; ++i) {
    for (j=0; j<T; ++j)
      printf("beta[%d][%d] = %f\t", i, j, fb->beta[i][j]);
    printf("\n");
  }

  printf("(forward-backward) LL=%g\n", LL);

  viterbi_example();

  srand(0); // make the randomized results reproducible
  bw_example();
  printf("\n\n");

  recognition_example();
  return 0;
}
