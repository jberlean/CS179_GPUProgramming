#ifndef CUDA_GENERAL_COAL_CU
#define CUDA_GENERAL_COAL_CU

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

// Device buffer variables
float* particle_vels[2]; 
float* particle_data[3]; 

#ifdef USE_FORCES_ARRAY
  float *forces;
#endif

__global__
void cudaInitKernel(float * vels_buffer, float * data_buffer1, float * data_buffer2, float * random, float box_width, 
                    float box_height, float min_vel, float max_vel, int num_particles)
{
  float *vels_x, *vels_y;
  float *pos_x, *pos_y;
  float *mass1, *mass2;

  vels_x = vels_buffer;
  vels_y = vels_buffer + num_particles;
  pos_x = data_buffer1;
  pos_y = data_buffer1 + num_particles;

  mass1 = data_buffer1 + 2 * num_particles;
  mass2 = data_buffer2 + 2 * num_particles;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num_particles)
  {
    int idx = 4 * i;
    // Randomly initialize velocities and positions. 
    vels_x[i] = min_vel + random[idx] * (max_vel - min_vel);
    vels_y[i] = min_vel + random[idx + 1] * (max_vel - min_vel);
    pos_x[i] = random[idx + 2] * box_width;
    pos_y[i] = random[idx + 3] * box_height;

    // Set mass to 1. 
    mass1[i] = 1;
    mass2[i] = 1;    

    i += blockDim.x * gridDim.x;
  }
}

void alloc_data() {
  // instantiate particle_vels, particle_data on GPU
  gpuErrChk(cudaMalloc((void **) &particle_vels[0], sizeof(float) * 2 * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_vels[1], sizeof(float) * 2 * num_particles));
  
  gpuErrChk(cudaMalloc((void **) &particle_data[0], sizeof(float) * 3 * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_data[1], sizeof(float) * 3 * num_particles));

  #ifdef USE_FORCES_ARRAY
    gpuErrChk(cudaMalloc((void **) &forces, 2 * sizeof(float) * num_particles));
  #endif
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

  // Rearrange data from the original interleaved format, to allow coalesced memory accesses
  float *temp_particle_data, *temp_particle_vels;
  temp_particle_data = new float[num_particles * 3];
  temp_particle_vels = new float[num_particles * 2];
  for (int i = 0; i < num_particles; i++) {
    temp_particle_data[i] = h_particle_data[3 * i];
    temp_particle_data[i + num_particles] = h_particle_data[3 * i + 1];
    temp_particle_data[i + 2 * num_particles] = h_particle_data[3 * i + 2];
    temp_particle_vels[i] = h_particle_vels[2 * i];
    temp_particle_vels[i + num_particles] = h_particle_vels[2 * i + 1];
  }
  gpuErrChk(cudaMemcpy(particle_data[0], temp_particle_data, 3 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(particle_data[1], temp_particle_data, 3 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(particle_vels[0], temp_particle_vels, 2 * num_particles * sizeof(float), cudaMemcpyHostToDevice));
  delete[] temp_particle_data;
  delete[] temp_particle_vels;
}

void delete_data() {
  // free all memory on GPU
  for (int i = 0; i < 2; i++)
  {
    gpuErrChk(cudaFree(particle_vels[i]));
    gpuErrChk(cudaFree(particle_data[i]));
  }

  #ifdef USE_FORCES_ARRAY
    gpuErrChk(cudaFree(forces));
  #endif
}

__device__
float2 get_force(float3 pos_data, float * data_old, int num_particles) {
  // sum force from every other particle based on mass, position of both particles
  float2 force = {0, 0};

  float3 other_data; // saves about 3s @ 128 threads/block and 1024 particles to store data_old[i], x_dist, and y_dist locally
  float x_dist, y_dist;

  float force_magnitude;
  for (int i = 0; i < num_particles; i++)
  {
    other_data.x = data_old[i];
    other_data.y = data_old[i + num_particles];
    other_data.z = data_old[i + 2 * num_particles];

    x_dist = pos_data.x - other_data.x;
    y_dist = pos_data.y - other_data.y;

    force_magnitude = pos_data.z * other_data.z * pow(x_dist * x_dist + y_dist * y_dist + SOFT_FACTOR, -1.5f);

    force.x -= x_dist * force_magnitude;
    force.y -= y_dist * force_magnitude;
  }
  return force;  
}

void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  float *temp_particle_data, *temp_particle_vels;
  temp_particle_data = new float[num_particles * 3];
  temp_particle_vels = new float[num_particles * 2];
  gpuErrChk(cudaMemcpy(temp_particle_data, particle_data[1 - pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(temp_particle_vels, particle_vels[1 - pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_particles; i++) {
    h_particle_data[3 * i] = temp_particle_data[i];
    h_particle_data[3 * i + 1] = temp_particle_data[i + num_particles];
    h_particle_data[3 * i + 2] = temp_particle_data[i + 2 * num_particles];
    h_particle_vels[2 * i] = temp_particle_vels[i];
    h_particle_vels[2 * i + 1] = temp_particle_vels[i + num_particles];
  }
  delete[] temp_particle_data;
  delete[] temp_particle_vels;
}

#endif // CUDA_GENERAL_COAL_CU
