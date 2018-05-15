#include <iostream>
#include <string>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
#define Sy 1
#define Sx 1
#endif

#ifndef Tnn
//Tiling Sizes
#define Tnn 32
#define Tn  16
#define Ti  16

#define Ty  8
#define Tx  8
#endif

#define NYPAD (Ny+Ky-1)
#define NXPAD (Nx+Kx-1)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];

VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_cuda)[NYSCL][NXSCL][Nn];

void fill_convolution_shared_simple(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                                    VTYPE (&neuron_i)[NYPAD][NXPAD][Ni]) {
  for (int yy = 0; yy < Ky; ++yy) {
    for (int xx = 0; xx < Kx; ++xx) {
      for (int nn = 0; nn < Nn; ++nn) {
        for (int ni = 0; ni < Ni; ++ni) {
          synapse[yy][xx][nn][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        }
      }
    }
  }
  for (int yy = 0; yy < NYPAD; ++yy) {
    for (int xx = 0; xx < NXPAD; ++xx) {
      for (int ni = 0; ni < Ni; ++ni) {
        neuron_i[yy][xx][ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
      }
    }
  }
}

std::pair<int, int> convolution_layer_blocked(
  VTYPE (&synapse)[Ky][Kx][Nn][Ni],
  VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
  VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  int c1 = 0, c2 = 0;
  VTYPE sum[Nn] = {0};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy / Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx / Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              for (int n = nn; n < nn + Tn; n++) {
                sum[n] = 0;
              }

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni - Ti + 1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc = 0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc += sv * nv;
                      }
                      sum[n] += sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++;
          }
          yout++;
        }
      }
    }
  }
}

void  convolution_layer(VTYPE (&synapse)[Ky][Kx][Nn][Ni],
                        VTYPE (&neuron_i)[NYPAD][NXPAD][Ni],
                        VTYPE (&neuron_n)[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn] = {0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n] = 0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n] += sv * nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++;
    }
    yout++;
  }
}

__global__ void convolution_layer_CUDA(VTYPE(&synapse)[Ky][Kx][Nn][Ni],
					   VTYPE(&neuron_i)[NYPAD][NXPAD][Ni],
					   VTYPE(&neuron_n)[NYSCL][NXSCL][Nn])
{
	if (blockIdx.x * 1024 + threadIdx.x < (Ny*Nx))
	{
		int index_x = ((blockIdx.x * 1024) + threadIdx.x) % Nx;
		int index_y = ((blockIdx.x * 1024) + threadIdx.x) / Ny;
		for (int n=0; n<Nn; n++)
		{
			VTYPE acc = 0;
			for (int x = index_x; x < index_x+3; x++)
			{
				for (int y = index_y; y < index_y+3; y++)
				{
					for (int z = 0; z<Ni; z++)
					{
						acc += neuron_i[y][x][z] * synapse[y - index_y][x - index_x][n][z];
					}
				}
			}
			neuron_n[index_y][index_x][n] = (acc < 0) ? (acc / 4) : acc;
		}
	}
}




int main(const int argc, const char** argv) {
  cout << "allocating memory\n";

  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])  aligned_malloc(64,  SYNAPSE_SIZE * sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni])aligned_malloc(64, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn])aligned_malloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));

  // Copy the data from the global buffers to the CUDA managed buffers.
  VTYPE(*neuron_i_cuda)[NYPAD][NXPAD][Ni];
  VTYPE(*neuron_n_cuda)[NYSCL][NXSCL][Nn];
  VTYPE(*synapse_cuda)[Ky][Kx][Nn][Ni];

  cudaMallocManaged(&neuron_i_cuda, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  cudaMallocManaged(&neuron_n_cuda, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  cudaMallocManaged(&synapse_cuda, SYNAPSE_SIZE * sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_convolution_shared_simple(*synapse, *neuron_i);

  for(int i=0; i<NYPAD; i++)
  {
	  for (int j = 0; j < NXPAD; j++)
	  {
		  for (int k = 0; k < Ni; k++)
		  {
			  (*neuron_i_cuda)[i][j][k] = (*neuron_i)[i][j][k];
		  }
	  }
  }

  for (int i = 0; i<Ky; i++)
  {
	  for (int j = 0; j < Kx; j++)
	  {
		  for (int k = 0; k < Ni; k++)
		  {
			  for (int l = 0; l < Nn; l++)
			  {
				  (*synapse_cuda)[i][j][k][l] = (*synapse)[i][j][k][l];
			  }
		  }
	  }
  }
  //cudaMemcpy(neuron_i_cuda, &neuron_i, sizeof(VTYPE), cudaMemcpyHostToDevice);
  //cudaMemcpy(synapse_cuda, &synapse, sizeof(VTYPE), cudaMemcpyHostToDevice);

  cout << "starting computation\n";

  //Simple Version
  begin_roi();
  convolution_layer(*synapse, *neuron_i, *neuron_n);
  end_roi();

  cout << "simple version complete!\n";


  //Blocked Version
  begin_roi();
  convolution_layer_blocked(*synapse, *neuron_i, *neuron_n2);
  end_roi();

  cout << "blocked computation complete!\n";
  compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n2, NYSCL * NXSCL * Nn);


  //Cuda version
  begin_roi();
  //TODO: Add cuda implementation of the layer.
  convolution_layer_CUDA <<<(222*222/1024+1), 1024>>> (*synapse_cuda, *neuron_i_cuda, *neuron_n_cuda);
  cudaDeviceSynchronize();
  end_roi();

  cout << "CUDA version complete!\n";
  compare((VTYPE*)*neuron_n, (VTYPE*)*neuron_n_cuda, NYSCL * NXSCL * Nn);

  cudaFree(neuron_i_cuda);
  cudaFree(synapse_cuda);
  cudaFree(neuron_n_cuda);

  cout << "done\n";
}


