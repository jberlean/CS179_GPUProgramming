#include <curand.h>
#include <cstdio>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "cuda_general_noncoal.cu"
#include "n_body_sim_cuda.cuh"


__global__
void interact_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  // each thread handles a particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < num_particles)
  {
    float2 force = get_force(data_old[i], data_old, num_particles);
    
    vels_new[i].x = vels_old[i].x + force.x * dt / data_old[i].z;
    vels_new[i].y = vels_old[i].y + force.y * dt / data_old[i].z;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

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


void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  gpuErrChk(cudaMemcpy(h_particle_data, particle_data[pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(h_particle_vels, particle_vels[pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));
}

std::string get_algorithm() {
  return std::string("SIMPLE");
}
