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
float2* particle_vels[2]; // x and y represent velocity in 2D
float3* particle_data[2]; // x and y represent position in 2D, z represents mass

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

void alloc_particle_info() {
  // instantiate particle_vels, particle_data on GPU
  gpuErrChk(cudaMalloc((void **) &particle_vels[0], sizeof(float2) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_vels[1], sizeof(float2) * num_particles));
  
  gpuErrChk(cudaMalloc((void **) &particle_data[0], sizeof(float3) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_data[1], sizeof(float3) * num_particles));
}

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block, int h_algorithm) 
{
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;
  algorithm = h_algorithm;

  // instantiate particle_vels, particle_data on GPU
  alloc_particle_info();
   
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
void init_data(int h_num_particles, float *h_particle_data, float *h_particle_vels, int h_num_blocks, int h_num_threads_per_block, int h_algorithm) {
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;
  algorithm = h_algorithm;

  alloc_particle_info();

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
}

__device__
float2 get_accel_lu1(float3 pos_data, float3 * data_old, int num_particles) {
  // sum accel due to every other particle based on mass, position of both particles
  float2 accel = {0, 0};

  float3 other_data1;
  float x_dist1, y_dist1;

  float accel_magnitude1;
  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i++)
  {
    other_data1 = data_old[i];
    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;

    accel_magnitude1 = other_data1.z / pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f);
    accel.x -= x_dist1 * accel_magnitude1;
    accel.y -= y_dist1 * accel_magnitude1;   
  }
  return accel;  
}

__device__
float2 get_accel_lu2(float3 pos_data, float3 * data_old, int num_particles) {
  // sum accel from every other particle based on mass, position of both particles
  float2 accel = {0, 0};

  float3 other_data1, other_data2;
  float x_dist1, y_dist1, x_dist2, y_dist2;

  float accel_magnitude1, accel_magnitude2;
  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i+=2)
  {
    other_data1 = data_old[i];
    other_data2 = data_old[i + 1];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;

    accel_magnitude1 = other_data1.z / pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f);
    accel_magnitude2 = other_data2.z / pow(x_dist2 * x_dist2 + y_dist2 * y_dist2 + soft_factor, 1.5f);

    accel.x -= x_dist1 * accel_magnitude1 + x_dist2 * accel_magnitude2;
    accel.y -= y_dist1 * accel_magnitude1 + y_dist2 * accel_magnitude2;
  }
  return accel;  
}

__device__
float2 get_accel_lu4(float3 pos_data, float3 * data_old, int num_particles) {
  // sum accel from every other particle based on mass, position of both particles
  float2 accel = {0, 0};

  float3 other_data1, other_data2, other_data3, other_data4;
  float x_dist1, y_dist1, x_dist2, y_dist2;
  float x_dist3, y_dist3, x_dist4, y_dist4;

  float accel_magnitude1, accel_magnitude2, accel_magnitude3, accel_magnitude4;
  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i+=4)
  {
    other_data1 = data_old[i];    
    other_data2 = data_old[i + 1];
    other_data3 = data_old[i + 2];
    other_data4 = data_old[i + 3];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;
    x_dist3 = pos_data.x - other_data3.x;
    y_dist3 = pos_data.y - other_data3.y;
    x_dist4 = pos_data.x - other_data4.x;
    y_dist4 = pos_data.y - other_data4.y;

    accel_magnitude1 = other_data1.z / pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f);
    accel_magnitude2 = other_data2.z / pow(x_dist2 * x_dist2 + y_dist2 * y_dist2 + soft_factor, 1.5f);
    accel_magnitude3 = other_data3.z / pow(x_dist3 * x_dist3 + y_dist3 * y_dist3 + soft_factor, 1.5f);
    accel_magnitude4 = other_data4.z / pow(x_dist4 * x_dist4 + y_dist4 * y_dist4 + soft_factor, 1.5f);

    accel.x -= x_dist1 * accel_magnitude1 + x_dist2 * accel_magnitude2 + 
               x_dist3 * accel_magnitude3 + x_dist4 * accel_magnitude4;
    accel.y -= y_dist1 * accel_magnitude1 + y_dist2 * accel_magnitude2 + 
               y_dist3 * accel_magnitude3 + y_dist4 * accel_magnitude4;
  }
  return accel;  
}

__device__
float2 get_accel_lu8(float3 pos_data, float3 * data_old, int num_particles) {
  // sum accel from every other particle based on mass, position of both particles
  float2 accel = {0, 0};

  float3 other_data1, other_data2, other_data3, other_data4;
  float3 other_data5, other_data6, other_data7, other_data8;

  float x_dist1, y_dist1, x_dist2, y_dist2;
  float x_dist3, y_dist3, x_dist4, y_dist4;
  float x_dist5, y_dist5, x_dist6, y_dist6;
  float x_dist7, y_dist7, x_dist8, y_dist8;

  float accel_magnitude1, accel_magnitude2, accel_magnitude3, accel_magnitude4;
  float accel_magnitude5, accel_magnitude6, accel_magnitude7, accel_magnitude8;

  float soft_factor = SOFT_FACTOR;
  for (int i = 0; i < num_particles; i+=8)
  {
    other_data1 = data_old[i];
    other_data2 = data_old[i + 1];
    other_data3 = data_old[i + 2];
    other_data4 = data_old[i + 3];
    other_data5 = data_old[i + 4];
    other_data6 = data_old[i + 5];
    other_data7 = data_old[i + 6];
    other_data8 = data_old[i + 7];

    x_dist1 = pos_data.x - other_data1.x;
    y_dist1 = pos_data.y - other_data1.y;
    x_dist2 = pos_data.x - other_data2.x;
    y_dist2 = pos_data.y - other_data2.y;
    x_dist3 = pos_data.x - other_data3.x;
    y_dist3 = pos_data.y - other_data3.y;
    x_dist4 = pos_data.x - other_data4.x;
    y_dist4 = pos_data.y - other_data4.y;
    x_dist5 = pos_data.x - other_data5.x;
    y_dist5 = pos_data.y - other_data5.y;
    x_dist6 = pos_data.x - other_data6.x;
    y_dist6 = pos_data.y - other_data6.y;
    x_dist7 = pos_data.x - other_data7.x;
    y_dist7 = pos_data.y - other_data7.y;
    x_dist8 = pos_data.x - other_data8.x;
    y_dist8 = pos_data.y - other_data8.y;

    accel_magnitude1 = other_data1.z / pow(x_dist1 * x_dist1 + y_dist1 * y_dist1 + soft_factor, 1.5f);
    accel_magnitude2 = other_data2.z / pow(x_dist2 * x_dist2 + y_dist2 * y_dist2 + soft_factor, 1.5f);
    accel_magnitude3 = other_data3.z / pow(x_dist3 * x_dist3 + y_dist3 * y_dist3 + soft_factor, 1.5f);
    accel_magnitude4 = other_data4.z / pow(x_dist4 * x_dist4 + y_dist4 * y_dist4 + soft_factor, 1.5f);
    accel_magnitude5 = other_data5.z / pow(x_dist5 * x_dist5 + y_dist5 * y_dist5 + soft_factor, 1.5f);
    accel_magnitude6 = other_data6.z / pow(x_dist6 * x_dist6 + y_dist6 * y_dist6 + soft_factor, 1.5f);
    accel_magnitude7 = other_data7.z / pow(x_dist7 * x_dist7 + y_dist7 * y_dist7 + soft_factor, 1.5f);
    accel_magnitude8 = other_data8.z / pow(x_dist8 * x_dist8 + y_dist8 * y_dist8 + soft_factor, 1.5f);

    accel.x -= x_dist1 * accel_magnitude1 + x_dist2 * accel_magnitude2 + 
               x_dist3 * accel_magnitude3 + x_dist4 * accel_magnitude4 +
               x_dist5 * accel_magnitude5 + x_dist6 * accel_magnitude6 + 
               x_dist7 * accel_magnitude7 + x_dist8 * accel_magnitude8;

    accel.y -= y_dist1 * accel_magnitude1 + y_dist2 * accel_magnitude2 + 
               y_dist3 * accel_magnitude3 + y_dist4 * accel_magnitude4 +
               y_dist5 * accel_magnitude5 + y_dist6 * accel_magnitude6 + 
               y_dist7 * accel_magnitude7 + y_dist8 * accel_magnitude8;
  }
  return accel;  
}


__device__
float2 get_accel(float3 pos_data, float3 * data_old, int num_particles) {
/*
  // sum accel from every other particle based on mass, position of both particles
#if UNROLLING == 1
  return get_accel_lu1(pos_data, data_old, num_particles);
#elif UNROLLING == 2
  return get_accel_lu2(pos_data, data_old, num_particles);
#elif UNROLLING == 4
  return get_accel_lu4(pos_data, data_old, num_particles);
#elif UNROLLING == 8
  return get_accel_lu8(pos_data, data_old, num_particles);
#else
  printf("Incorrect unrolling factor given %d", UNROLLING);
#endif
*/
  float2 f = {0,0};
  return f;
}


__global__
void simple_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  // each thread handles a particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < num_particles)
  {
    float2 accel = get_accel(data_old[i], data_old, num_particles);
    
    vels_new[i].x = vels_old[i].x + accel.x * dt;
    vels_new[i].y = vels_old[i].y + accel.y * dt;
    
    data_new[i].x = data_old[i].x + vels_new[i].x * dt; 
    data_new[i].y = data_old[i].y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

__global__
void pxp_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  extern __shared__ float3 sdata[];
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  while (i < num_particles)
  {
    float2 accel;
    accel.x = 0;
    accel.y = 0; 

    float3 pos_data = data_old[i];
    // NOTE: num_particles is a multiple of num_threads_per_block.
    for (int num_tile = 0; num_tile * blockDim.x < num_particles; num_tile++)
    {
      __syncthreads();
      sdata[tid] = data_old[num_tile * blockDim.x + tid];
      __syncthreads();
      float2 block_accel = get_accel(pos_data, sdata, blockDim.x);
      accel.x += block_accel.x;
      accel.y += block_accel.y;
    }    
    
    vels_new[i].x = vels_old[i].x + accel.x * dt;
    vels_new[i].y = vels_old[i].y + accel.y * dt;
    
    data_new[i].x = pos_data.x + vels_new[i].x * dt; 
    data_new[i].y = pos_data.y + vels_new[i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

__global__
void pxp_opt_kernel(float2 * vels_old, float2 * vels_new, float3 * data_old, float3 * data_new, float dt, int num_particles) {
  extern __shared__ float3 sdata[];
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  float2 vel_new;
  
  while (i < num_particles)
  {
    float2 accel;
    accel.x = 0;
    accel.y = 0; 

    float3 pos_data = data_old[i];
    // NOTE: num_particles is a multiple of num_threads_per_block.
    for (int num_tile = 0; num_tile * blockDim.x < num_particles; num_tile++)
    {
      __syncthreads();
      sdata[tid] = data_old[num_tile * blockDim.x + tid];
      __syncthreads();
      float2 block_accel = get_accel(pos_data, sdata, blockDim.x);
      accel.x += block_accel.x;
      accel.y += block_accel.y;
    }    
    
    vel_new.x = vels_old[i].x + accel.x * dt;
    vel_new.y = vels_old[i].y + accel.y * dt;
    
    data_new[i].x = pos_data.x + vel_new.x * dt; 
    data_new[i].y = pos_data.y + vel_new.y * dt;

    i += blockDim.x * gridDim.x;
  }
}
 
void call_interact_kernel(float dt) {
  // call kernel
  if (algorithm == SIMPLE)
  {
    simple_kernel<<<num_blocks, num_threads_per_block>>>(particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  }
  else if (algorithm == PXP)
  {
    pxp_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float3)>>>
                                                        (particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  }
  else if (algorithm == PXP_OPT)
  {
    pxp_opt_kernel<<<num_blocks, num_threads_per_block, num_threads_per_block * sizeof(float3)>>>
                                                        (particle_vels[pingpong], particle_vels[1 - pingpong], 
                                                           particle_data[pingpong], particle_data[1 - pingpong], 
                                                           dt, num_particles);
  } else {
    std::cout << "Invalid algorithm supplied: " << algorithm << "\n";
  }

  // update pingpong
  pingpong = 1 - pingpong;
}


void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  gpuErrChk(cudaMemcpy(h_particle_data, particle_data[1 - pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(h_particle_vels, particle_vels[1 - pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));
}
