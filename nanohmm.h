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

// The interface for the nanohmm library.
#ifndef NANOHMM_H__
#define NANOHMM_H__

struct hmm_s {
  unsigned int N;      // number of states
  unsigned int M;      // number of possible observations
  double **A;          // N x N state transition matrix
  double **B;          // N x M observation probability matrix
  double *pi;          // N-element vector for initial state probabilities
};
typedef struct hmm_s hmm_t;


struct forward_s {
  const hmm_t *lambda;
  double **alpha;       // N x T array for the forward variable alpha
  double *c;            // N-element vector of re-normalization coefficients
};
typedef struct forward_s forward_t;


struct forwardbackward_s {
  const hmm_t *lambda;
  double **alpha;       // N x T array for the forward variable alpha
  double *c;            // N-element vector of re-normalization coefficients
  double **beta;        // N x T array for the backward variable beta
};
typedef struct forwardbackward_s forwardbackward_t;


struct viterbi_s {
  const hmm_t *lambda;
  double *delta;
  double *delta_next;
  unsigned int **psi;
};
typedef struct viterbi_s viterbi_t;


struct baumwelch_s {
  hmm_t *lambda;  // M and N need to be set
  double **alpha;
  double *c;
  double **beta;
  double *sgamma; // sum of all gamma's for a given state over time
};
typedef struct baumwelch_s baumwelch_t;



#ifdef __cplusplus
extern "C" {
#endif


  // Returns the log-likelihood (base 2) for the sequence O of length T.
  double forward(forward_t *f,
                 const unsigned int *O,
                 const unsigned int T);


  // Runs the forward-backward algorithm on a sequence O of length T.
  // Runs the forward algorithm to compute the alphas and then runs the backward
  // algorithm to compute the betas. Returns the log-likelihood (base 2) of the
  // observation sequence computed using the forward algorithm.
  double forwardbackward(forwardbackward_t *fb,
                         const unsigned int *O,
                         const unsigned int T);


  // Runs the Viterbi algorithm on the observation sequence O of length T.
  // The output is stored in the array Q, which needs to be large enough to
  // hold the T elements of the generated sequence. The function returns the
  // log-likelihood (logarithm base 2) of the most likely state sequence Q.
  double viterbi(viterbi_t *v, const unsigned int *O, const unsigned int T,
                 unsigned int *Q);


  // Runs the Baum-Welch algorithm for num_iter iterations to train the HMM model.
  // Returns the log-likelihood (logarithm 2) of the observation sequence O
  // of length T, given the trained model bw->lambda.
  double baumwelch(baumwelch_t *bw,
                   const unsigned int *O,
                   const unsigned int T,
                   unsigned int num_iter);


  // Returns the number of bytes necessary to initialize the hmm_t structure as
  // a single contiguous memory block.
  static inline unsigned int hmm_block_size(unsigned int N, unsigned int M) {
    return sizeof(hmm_t) + 2*N*sizeof(double*) + N*(N + M + 1)*sizeof(double);
  }

  // Returns the number of bytes necessary to initialize the forward_t structure
  // as a single contiguous memory block.
  static inline unsigned int forward_block_size(unsigned int N,
                                                unsigned int T) {
    return sizeof(forward_t) + N*sizeof(double*)
      + (N+1)*T*sizeof(double);
  }

  // Returns the number of bytes necessary to initialize the forwardbackward_t
  // structure as a single contiguous memory block.
  static inline unsigned int forwardbackward_block_size(unsigned int N,
                                                        unsigned int T) {
    return sizeof(forwardbackward_t) + 2*N*sizeof(double*)
      + (2*N+1)*T*sizeof(double);
  }

  // Computes the minimum number of bytes necessary to initialize the viterbi_t
  // structure as a single contiguous memory block.
  static inline unsigned int viterbi_block_size(unsigned int N, unsigned int T) {
    return sizeof(viterbi_t) + 2*N*sizeof(double) + N*sizeof(unsigned int*)
      + N*T*sizeof(unsigned int);
  }

  // Returns the number of bytes necessary to initialize the baumwelch_t
  // structure as a single contiguous memory block.
  static inline unsigned int baumwelch_block_size(unsigned int N,
                                                  unsigned int M,
                                                  unsigned int T) {
    return sizeof(baumwelch_t) + 2*N*sizeof(double*) + (3*N+1)*T*sizeof(double);
  }

  // Initializes the internal pointers for an hmm_t structure that uses a single
  // contiguous memory block.
  hmm_t* hmm_init_block(void *block, unsigned int N, unsigned int M);

  // Initializes the internal pointers for a forward_t structure that uses a
  // single contiguous memory block. The parameter hsize specifies the offset in
  // bytes relative to the block's start pointer where the additional data can
  // be stored. If hsize is zero, then it is set to sizeof(forward_t).
  forward_t* forward_init_block(void *block, const hmm_t *lambda,
                                unsigned int T, unsigned int hsize);

  // Initializes the internal pointers for a forwardbackward_t structure that
  // uses a single contiguous memory block. The parameter hsize specifies the
  // offset in bytes relative to the block's start pointer where the additional
  // data can be stored. If hsize is zero, then it it is assumed that this
  // offset is sizeof(forwardbackward_t).
  forwardbackward_t* forwardbackward_init_block(void *block,
                                                const hmm_t *lambda,
                                                const unsigned int T,
                                                unsigned int hsize);

  // Initializes the internal pointers for a viterbi_t structure that uses a
  // single contiguous memory block. The parameter hsize specifies the offset in
  // bytes relative to the block's start pointer where the additional data can
  // be stored. If hsize is zero, then the offset is set to sizeof(viterbi_t).
  viterbi_t* viterbi_init_block(void *block,
                                const hmm_t *lambda,
                                const unsigned int T,
                                unsigned int hsize);

  // Initializes the internal pointers for a baumwelch_t structure that uses a
  // single contiguous memory block. The parameter hsize specifies the offset in
  // bytes relative to the block's start pointer where the additional data can
  // be stored. If hsize is zero, then it is set to sizeof(baumwelch_t).
  baumwelch_t* baumwelch_init_block(void *block,
                                    const hmm_t *lambda,
                                    const unsigned int T,
                                    unsigned int hsize);

#ifdef __cplusplus
}
#endif


#endif
