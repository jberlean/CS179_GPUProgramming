#ifndef CUDA_GENERAL_NONCOAL_CUH
#define CUDA_GENERAL_NONCOAL_CUH

#include <curand.h>
#include <cstdio>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

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


// Flag for pingpong;
int pingpong = 0;

// Number particles; determined at runtime.
int num_particles;    
 
int num_blocks;
int num_threads_per_block;


// Device buffer variables
float2* particle_vels[2]; // x and y represent velocity in 2D
float3* particle_data[2]; // x and y represent position in 2D, z represents mass

__global__
void cudaInitKernel(float2 * vels_buffer, float3 * data_buffer1, float3 * data_buffer2, float * random, float box_width, 
                    float box_height, float min_vel, float max_vel, int num_particles);

void alloc_data();

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block);
void init_data(int h_num_particles, float *h_particle_data, float *h_particle_vels, int h_num_blocks, int h_num_threads_per_block)

void delete_data();

__device__
float2 get_force(float3 pos_data, float3 * data_old, int num_particles);

void get_particle_data(float * h_particle_data, float * h_particle_vels);

#endif // CUDA_GENERAL_NONCOAL_CUH