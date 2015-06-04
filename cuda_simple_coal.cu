#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

__global__
void interact_kernel(float * vels_old, float * vels_new, float * data_old, float * data_new, float dt, int num_particles) {
  // each thread handles a particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (i < num_particles)
  {
    float3 pos_data;
    pos_data.x = data_old[i];
    pos_data.y = data_old[i + num_particles];
    pos_data.z = data_old[i + 2 * num_particles];
    
    float2 force = get_force(pos_data, data_old, num_particles);
    
    vels_new[i] = vels_old[i] + force.x * dt / data_old[i + 2 * num_particles];
    vels_new[i + num_particles] = vels_old[i + num_particles] + force.y * dt / data_old[i + 2 * num_particles];
    
    data_new[i] = data_old[i] + vels_new[i] * dt; 
    data_new[i + num_particles] = data_old[i + num_particles] + vels_new[i + num_particles] * dt;

    i += blockDim.x * gridDim.x;
  }
}
 
void simulate_time_step(float dt) {
  // call kernel
  interact_kernel<<<num_blocks, num_threads_per_block>>>(particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  
  // update pingpong
  pingpong = 1 - pingpong;
}

std::string get_algorithm() {
  return std::string("SIMPLE_COALESCED");
}
