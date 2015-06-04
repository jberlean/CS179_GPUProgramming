#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

#include "cuda_general_noncoal.cu"

__global__
void interact_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  extern __shared__ float3 sdata[];
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  while (i < num_particles)
  {
    float2 force = {0, 0};

    float3 pos_data = data_old[i];
    // NOTE: num_particles is a multiple of num_threads_per_block.
    for (int num_tile = 0; num_tile * blockDim.x < num_particles; num_tile++)
    {
      __syncthreads();
      sdata[tid] = data_old[num_tile * blockDim.x + tid];
      __syncthreads();
      float2 block_force = get_force(pos_data, sdata, blockDim.x);
      force.x += block_force.x;
      force.y += block_force.y;
    }    
    
    vels_new[i].x = vels_old[i].x + force.x * dt / data_old[i].z; // TODO: replace data_old[i] with pos_data
    vels_new[i].y = vels_old[i].y + force.y * dt / data_old[i].z;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

 
void simulate_time_step(float dt) {
  // call kernel
  interact_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float3)>>>
                                                        (particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  
  // update pingpong
  pingpong = 1 - pingpong;
}


std::string get_algorithm() {
  return std::string("PxP");
}