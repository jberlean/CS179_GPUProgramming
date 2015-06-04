#define USE_FORCES_ARRAY

#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

#include "cuda_general_coal.cu"

__global__
void calc_forces_kernel(float * forces, float * vels_old, float * vels_new, float * data_old, 
                           float * data_new, float dt, int num_particles) 
{
  extern __shared__ float sdata[];
  
  int tile_id = blockIdx.x;
  int tid = threadIdx.x;
  
  int num_tiles_per_col = num_particles / blockDim.x;
  int num_tiles = num_particles * num_particles / (blockDim.x * blockDim.x);

  while (tile_id < num_tiles)
  {
    int rid = (tile_id % num_tiles_per_col) * blockDim.x + tid;
    int cid = (tile_id/num_tiles_per_col) * blockDim.x + tid;
    
    sdata[tid] = data_old[cid];
    sdata[tid + blockDim.x] = data_old[cid + num_particles];
    sdata[tid + 2 * blockDim.x] = data_old[cid + 2 * num_particles];
 
    __syncthreads();

    float3 pos_data;
    pos_data.x = data_old[rid];
    pos_data.y = data_old[rid + num_particles];
    pos_data.z = data_old[rid + 2 * num_particles];

    float2 block_force = get_force(pos_data, sdata, blockDim.x);
    atomicAdd(forces + rid, block_force.x);
    atomicAdd(forces + rid + num_particles, block_force.y);
   
    __syncthreads();


    tile_id += gridDim.x;
  }
}

__global__
void apply_forces_kernel(float * forces, float * vels_old, float * vels_new, float * data_old, 
                         float * data_new, float dt, int num_particles)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (i < num_particles)
  {
    float2 force;
    force.x = forces[i];
    force.y = forces[i + num_particles];

    vels_new[i] = vels_old[i] + force.x * dt / data_old[i + 2 * num_particles]; // TODO: replace data_old[i] with pos_data
    vels_new[i + num_particles] = vels_old[i + num_particles] + force.y * dt / data_old[i + 2 * num_particles];
    
    data_new[i] = data_old[i] + vels_new[i] * dt; 
    data_new[i + num_particles] = data_old[i + num_particles] + vels_new[i + num_particles] * dt;

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
  return std::string("PxP_OPT_COALESCED");
}
