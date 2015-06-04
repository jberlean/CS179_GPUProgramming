#include <curand.h>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>

#include "n_body_sim_cuda.cuh"

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

// Algorithm to use.
int algorithm;

// Device buffer variables
float *particle_vels[2]; 
float *particle_data[2];
float *forces;

__global__
void cudaInitKernel(float * vels_buffer, float * data_buffer1, float * data_buffer2, float * random, float box_width, 
                    float box_height, float min_vel, float max_vel, int num_particles)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num_particles)
  {
    vels_buffer[i] = min_vel + random[4 * i] * (max_vel - min_vel);
    vels_buffer[i + num_particles] = min_vel + random[4 * i + 1] * (max_vel - min_vel);
    data_buffer1[i] = random[4 * i + 2] * box_width;
    data_buffer1[i + num_particles] = random[4 * i + 3] * box_height;
    data_buffer1[i + 2 * num_particles] = 1;

    data_buffer2[i + 2 * num_particles] = 1;    

/*    if (i == 0) {
      data_buffer1[i].z = 1000;
      data_buffer2[i].z = 1000;
      data_buffer1[i].x = box_width / 2;
      data_buffer1[i].y = box_height / 2;
      vels_buffer[i].x = 0;
      vels_buffer[i].y = 0;
    }
*/

    i += blockDim.x * gridDim.x;
  }
}

void alloc_data() {
  // instantiate particle_vels, particle_data on GPU
  gpuErrChk(cudaMalloc((void **) &particle_vels[0], sizeof(float) * 2 * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_vels[1], sizeof(float) * 2 * num_particles));
  
  gpuErrChk(cudaMalloc((void **) &particle_data[0], sizeof(float) * 3 * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_data[1], sizeof(float) * 3 * num_particles));

  gpuErrChk(cudaMalloc((void **) &forces, 2 * sizeof(float) * num_particles));
}

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block) 
{
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;

  // instantiate particle_vels, particle_data on GPU
  alloc_data();
   
  // set initial values for particle_vels, particle_data on GPU
  float * random;
  gpuErrChk(cudaMalloc((void **) &random, sizeof(float) * num_particles * 4));   
  
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, random, num_particles * 4);

  cudaInitKernel<<<num_blocks, num_threads_per_block>>>(particle_vels[0], particle_data[0], particle_data[1], 
                                                        random, box_width, box_height, min_vel, max_vel, num_particles);

  curandDestroyGenerator(gen);
  gpuErrChk(cudaFree(random));
}
void init_data(int h_num_particles, float *h_particle_data, float *h_particle_vels, int h_num_blocks, int h_num_threads_per_block) {
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;

  alloc_data();

  // TODO: Change this for memory coalescing
  gpuErrChk(cudaMemcpy(particle_data[0], h_particle_data, 3 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(particle_data[1], h_particle_data, 3 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(particle_vels[0], h_particle_vels, 2 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
}

void delete_data() {
  // free all memory on GPU
  for (int i = 0; i < 2; i++)
  {
    gpuErrChk(cudaFree(particle_vels[i]));
    gpuErrChk(cudaFree(particle_data[i]));
  }

  gpuErrChk(cudaFree(forces));
}

__device__
float2 get_force(float3 pos_data, float * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force = {0, 0};

  float3 other_data1; // saves about 3s @ 128 threads/block and 1024 particles to store data_old[i], x_dist, and y_dist locally
  float x_dist1, y_dist1;

  float force_magnitude1;
  for (int i = 0; i < num_particles; i++)
  {
    other_data1 = data_old[i];
    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;

    force_magnitude1 = pos_data.z * other_data1.z * pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + SOFT_FACTOR, -1.5f);

    force.x -= x_dist1 * force_magnitude1;
    force.y -= y_dist1 * force_magnitude1;
  }
  return force;
}

__global__
void calc_forces_kernel(float * forces, float * vels_old, float * vels_new, float * data_old, float * data_new, float dt, int num_particles) {
  extern __shared__ float sdata[];
  
  int tile_id = blockIdx.x;
  int tid = threadIdx.x;
  
  int num_tiles_per_col = num_particles / blockDim.x;
  int num_tiles = num_particles * num_particles / (blockDim.x * blockDim.x);

  float3 pos_data;
  
  pos_data.x = data_old[rid];
  pos_data.y = data_old[rid + num_particles];
  pos_data.z = data_old[rid + 2 * num_particles];

  while (tile_id < num_tiles)
  {
    int rid = (tile_id % num_tiles_per_col) * blockDim.x + tid;
    int cid = (tile_id/num_tiles_per_col) * blockDim.x + tid;

    sdata[tid] = data_old[cid];
    sdata[tid + blockDim.x] = data_old[cid + num_particles];
    sdata[tid + 2 * blockDim.x] = data_old[cid + 2 * num_particles];
 
    __syncthreads();

    float2 block_force = get_force(pos_data, sdata, blockDim.x);
    atomicAdd(forces + rid, block_force.x);
    atomicAdd(forces + rid + num_particles, block_force.y);
    
    __syncthreads();

    tile_id += gridDim.x;

    // Check if need to reload pos_data from global memory
    if (gridDim.x % num_tiles_per_col == 0) {
      pos_data.x = data_old[rid];
      pos_data.y = data_old[rid + num_particles];
      pos_data.z = data_old[rid + 2 * num_particles];
    }
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

  gpuErrChk(cudaMemset(forces, 0, num_particles * 2 * sizeof(float)));

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


void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  gpuErrChk(cudaMemcpy(h_particle_data, particle_data[1 - pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(h_particle_vels, particle_vels[1 - pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));
}


std::string get_algorithm() {
  return std::string("PxP_OPT");
}