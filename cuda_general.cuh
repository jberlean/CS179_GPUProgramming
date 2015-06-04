#ifndef CUDA_GENERAL_CUH
#define CUDA_GENERAL_CUH

#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

// GENERAL UTILITY FUNCTIONS USED BY MULTIPLE ALGORITHMS


// macro for error-handling
#define gpuErrChk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


void alloc_particle_info(float **particle_data, float **particle_vels);

__global__
void cudaInitKernel_uncoalesced(float2 * vels_buffer, float3 * data_buffer1, float3 * data_buffer2, float * random, float box_width, 
                    float box_height, float min_vel, float max_vel, int num_particles);


void init_data_uncoalesced(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block);

void init_data_uncoalesced(int h_num_particles, float *h_particle_data, float *h_particle_vels, int h_num_blocks, int h_num_threads_per_block);

void delete_data_uncoalesced(float **particle_data, float **particle_vels);

void get_particle_data_uncoalesced(float * h_particle_data, float * h_particle_vels);

#endif //CUDA_GENERAL_CUH