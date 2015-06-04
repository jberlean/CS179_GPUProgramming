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
 
// Parameters for the kernel.
int num_blocks;
int num_threads_per_block;

// Device buffer variables.
float* particle_vels[2]; 
float* particle_data[3]; 

// Used for PXPOPT algorithm.
#ifdef USE_ACCEL_ARRAY
  float *accel;
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

  #ifdef USE_ACCEL_ARRAY
    gpuErrChk(cudaMalloc((void **) &accel, 2 * sizeof(float) * num_particles));
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

  // Rearrange data from the original interleaved format to allow coalesced memory accesses.
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

  #ifdef USE_ACCEL_ARRAY
    gpuErrChk(cudaFree(accel));
  #endif
}

__device__
float2 get_accel(float3 pos_data, float * data_old, int num_particles) {
  // sum acceleration from every other particle based on mass, position of both particles
  float2 accel = {0, 0};

  float3 other_data1, other_data2, other_data3, other_data4;
  float x_dist1, x_dist2, x_dist3, x_dist4;
  float y_dist1, y_dist2, y_dist3, y_dist4;

  float *pos_x = data_old;
  float *pos_y = data_old + num_particles;
  float *mass = data_old + 2 * num_particles;

  float accel_mag1, accel_mag2, accel_mag3, accel_mag4;
  for (int i = 0; i < num_particles; i += 4)
  {
    other_data1.x = pos_x[i];
    other_data1.y = pos_y[i];
    other_data1.z = mass[i];
    other_data2.x = pos_x[i+1];
    other_data2.y = pos_y[i+1];
    other_data2.z = mass[i+1];
    other_data3.x = pos_x[i+2];
    other_data3.y = pos_y[i+2];
    other_data3.z = mass[i+2];
    other_data4.x = pos_x[i+3];
    other_data4.y = pos_y[i+3];
    other_data4.z = mass[i+3];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;
    x_dist3 = pos_data.x - other_data3.x;
    y_dist3 = pos_data.y - other_data3.y;
    x_dist4 = pos_data.x - other_data4.x;
    y_dist4 = pos_data.y - other_data4.y;

    accel_mag1 = other_data1.z * pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + SOFT_FACTOR, -1.5f);
    accel_mag2 = other_data2.z * pow(x_dist2 * x_dist2 + y_dist2 * y_dist2 + SOFT_FACTOR, -1.5f);
    accel_mag3 = other_data3.z * pow(x_dist3 * x_dist3 + y_dist3 * y_dist3 + SOFT_FACTOR, -1.5f);
    accel_mag4 = other_data4.z * pow(x_dist4 * x_dist4 + y_dist4 * y_dist4 + SOFT_FACTOR, -1.5f);

    accel.x -= fma(x_dist1, accel_mag1, fma(x_dist2, accel_mag2, fma(x_dist3, accel_mag3, x_dist4 * accel_mag4)));
    accel.y -= fma(y_dist1, accel_mag1, fma(y_dist2, accel_mag2, fma(y_dist3, accel_mag3, y_dist4 * accel_mag4)));
  }
  return accel;  
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
