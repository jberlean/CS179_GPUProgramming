#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

#include "cuda_general_coal.cu"

__global__
void interact_kernel(float * vels_old, float * vels_new, float * data_old, float * data_new, float dt, int num_particles) {
  extern __shared__ float sdata[];
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  while (i < num_particles)
  {
    float2 acc = {0, 0};

    float3 pos_data;
    pos_data.x = data_old[i];
    pos_data.y = data_old[i + num_particles];
    pos_data.z = data_old[i + 2 * num_particles];

    // NOTE: num_particles is a multiple of num_threads_per_block.
    for (int num_tile = 0; num_tile * blockDim.x < num_particles; num_tile++)
    {
      __syncthreads();
      sdata[tid] = data_old[num_tile * blockDim.x + tid];
      sdata[tid + blockDim.x] = data_old[num_tile * blockDim.x + tid + num_particles];
      sdata[tid + 2 * blockDim.x] = data_old[num_tile * blockDim.x + tid + 2 * num_particles];
 
      __syncthreads();
      float2 block_accel = get_accel(pos_data, sdata, blockDim.x);
      acc.x += block_accel.x;
      acc.y += block_accel.y;
    }    
    
    vels_new[i] = vels_old[i] + acc.x * dt;
    vels_new[i + num_particles] = vels_old[i + num_particles] + acc.y * dt;
    
    data_new[i] = data_old[i] + vels_new[i] * dt; 
    data_new[i + num_particles] = data_old[i + num_particles] + vels_new[i + num_particles] * dt;

    i += blockDim.x * gridDim.x;
  }
}
 
void simulate_time_step(float dt) {
  // call kernel
  interact_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float) * 3>>>(particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  
  // update pingpong
  pingpong = 1 - pingpong;
}

std::string get_algorithm() {
  return std::string("PxP_COALESCED");
}
