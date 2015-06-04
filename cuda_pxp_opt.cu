#define USE_FORCES_ARRAY

#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

#include "cuda_general_noncoal.cu"

__global__
void calc_forces_kernel(float2 *forces, float2 *vels_old, float2 *vels_new, float3 *data_old, float3 *data_new, float dt, int num_particles) {
  extern __shared__ float3 sdata[];
  
  int tile_id = blockIdx.x;
  int tid = threadIdx.x;
  
  int num_tiles_per_col = num_particles / blockDim.x;
  int num_tiles = num_particles * num_particles / (blockDim.x * blockDim.x);

  while (tile_id < num_tiles)
  {
    int rid = (tile_id % num_tiles_per_col) * blockDim.x + tid;
    int cid = (tile_id/num_tiles_per_col) * blockDim.x + tid;
    
    sdata[tid] = data_old[cid];
 
    __syncthreads();

    float2 block_force = get_force(data_old[rid], sdata, blockDim.x);
    atomicAdd(&forces[rid].x, block_force.x);
    atomicAdd(&forces[rid].y, block_force.y);
    
    __syncthreads();

    tile_id += gridDim.x;
  }
}

__global__
void apply_forces_kernel(float2 *forces, float2 *vels_old, float2 *vels_new, float3 *data_old, 
                         float3 *data_new, float dt, int num_particles)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (i < num_particles)
  {
    float2 force = forces[i];
    
    vels_new[i].x = vels_old[i].x + force.x * dt / data_old[i].z; 
    vels_new[i].y = vels_old[i].y + force.y * dt / data_old[i].z;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

void simulate_time_step(float dt) {
  // call kernel

  gpuErrChk(cudaMemset(forces, 0, num_particles * sizeof(float2)));

  calc_forces_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float) * 3>>>
                                                       (forces, particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                         particle_data[pingpong], particle_data[1 - pingpong], 
                                                         dt, num_particles);
  
  apply_forces_kernel<<<num_blocks, num_threads_per_block>>>(forces, particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                         particle_data[pingpong], particle_data[1 - pingpong], 
                                                         dt, num_particles);

  // update pingpong
  pingpong = 1 - pingpong;
}


std::string get_algorithm() {
  return std::string("PxP_OPT");
}
