#include <iostream>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
#define Nn 128  // Number of Output Layers
#define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
// Tiling Sizes
#define Tnn 32
#define Tii 32
//#define Tn 5
//#define Ti 25
#define Tn 16
#define Ti 16
#endif

//Arrays:
VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64))),    neuron_n2[Nn] __attribute__((aligned(64)));

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni],
                     VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_n2)[Nn]) {
  for (int n = 0; n < Nn; ++n) {
    for (int i = 0; i < Ni; ++i) {
      synapse[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for (int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for (int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0; //i;
    neuron_n2[n] = 0; //i;
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  int total_calc = 0;
  for (int n = 0; n < Nn; n++) {
    VTYPE temp = 0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

void classifier_layer_blocked(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni],
                              VTYPE (&neuron_n)[Nn]) {
  int total_calc = 0;
  VTYPE sum[Nn] = {0};
  for (int nnn = 0; nnn < Nn; nnn += Tnn) { // tiling for output neurons;
    for (int iii = 0; iii < Ni; iii += Tii) { // tiling for input neurons;
      for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
        for (int ii = iii; ii < iii + Tii; ii += Ti) {
          // — Original code —
          for (int n = nn; n < nn + Tn; n++) {
            VTYPE sum_sc = 0;
            for (int i = ii; i < ii + Ti; i++) {
              sum_sc += (synapse[n][i] * neuron_i[i]);
            }
            sum[n] += sum_sc;
          }
        }
      }
    }
    for (int nn = nnn; nn < nnn + Tnn; nn++) {
      neuron_n[nn] = transfer(sum[nn]);
    }
  }
}

__global__ void classifier_layer_kernel(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni],
                              VTYPE (&neuron_n)[Nn]) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  VTYPE acc = 0;
  for (int i = 0; i < Ni; i++) {
    acc += synapse[tid][i] * neuron_i[i];
  }
  acc = (acc > 0) ? acc : acc / 4;
  neuron_n[tid] = acc;

}

void classifier_layer_cuda(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni],
                              VTYPE (&neuron_n)[Nn]) {
  classifier_layer_kernel<<<Nn / 1024, 1024>>>(synapse, neuron_i, neuron_n);
  cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
  cout << "initializing arrays\n";

  fill_classifier(synapse, neuron_i, neuron_n, neuron_n2);

  // Copy the data from the global buffers to the CUDA managed buffers.
  VTYPE (*neuron_i_cuda)[Ni];
  VTYPE (*neuron_n_cuda)[Nn];
  VTYPE (*synapse_cuda)[Nn][Ni];

  cudaMallocManaged(&neuron_i_cuda, sizeof(VTYPE) * Ni);
  cudaMallocManaged(&neuron_n_cuda, sizeof(VTYPE) * Nn);
  cudaMallocManaged(&synapse_cuda, sizeof(VTYPE) * Nn * Ni);

  for (int i = 0; i < Ni; i++) {
    (*neuron_i_cuda)[i] = neuron_i[i];
  }

  for (int i = 0; i < Nn; i++) {
    (*neuron_n_cuda)[i] = neuron_n[i];
  }

  for (int i = 0; i < Nn; i++) {
    for (int j = 0; j < Ni; j++) {
      (*synapse_cuda)[i][j] = synapse[i][j];
    }
  }

  cout << "starting computation\n";

  begin_roi();
  classifier_layer(synapse, neuron_i, neuron_n);
  end_roi();

  cout << "simple version complete!\n";

  begin_roi();
  classifier_layer_blocked(synapse, neuron_i, neuron_n2);
  end_roi();

  cout << "blocked computation complete!\n";

  compare(neuron_n, neuron_n2, Nn);

  // CUDA implemetation
  begin_roi();
  classifier_layer_cuda(*synapse_cuda, *neuron_i_cuda, *neuron_n_cuda);
  end_roi();

  cout << "cuda computation complete!\n";

  compare(neuron_n, *neuron_n_cuda, Nn);

  cout << "done\n";
}

