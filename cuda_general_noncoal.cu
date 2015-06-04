#ifndef CUDA_GENERAL_NONCOAL_CU
#define CUDA_GENERAL_NONCOAL_CU

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

#ifdef USE_FORCES_ARRAY
  float2 *forces;
#endif

__global__
void cudaInitKernel(float2 * vels_buffer, float3 * data_buffer1, float3 * data_buffer2, float * random, float box_width, 
                    float box_height, float min_vel, float max_vel, int num_particles)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num_particles)
  {
    vels_buffer[i].x = min_vel + random[4 * i] * (max_vel - min_vel);
    vels_buffer[i].y = min_vel + random[4 * i + 1] * (max_vel - min_vel);
    data_buffer1[i].x = random[4 * i + 2] * box_width;
    data_buffer1[i].y = random[4 * i + 3] * box_height;
    data_buffer1[i].z = 1;

    data_buffer2[i].z = 1;    

    i += blockDim.x * gridDim.x;
  }
}

void alloc_data() {
  // instantiate particle_vels, particle_data on GPU
  gpuErrChk(cudaMalloc((void **) &particle_vels[0], sizeof(float2) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_vels[1], sizeof(float2) * num_particles));
  
  gpuErrChk(cudaMalloc((void **) &particle_data[0], sizeof(float3) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_data[1], sizeof(float3) * num_particles));

  #ifdef USE_FORCES_ARRAY
    gpuErrChk(cudaMalloc((void **) &forces, sizeof(float2) * num_particles));
  #endif
}

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block) 
{
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;
  pingpong = 0;

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
  pingpong = 0;

  alloc_data();

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

  #ifdef USE_FORCES_ARRAY
    gpuErrChk(cudaFree(forces));
  #endif
}

__device__
float2 get_force(float3 pos_data, float3 * data_old, int num_particles) {
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

void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  gpuErrChk(cudaMemcpy(h_particle_data, particle_data[pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(h_particle_vels, particle_vels[pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));
}

#endif // CUDA_GENERAL_NONCOAL_CU