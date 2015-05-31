// -*- C++ -*-
#ifndef N_BODY_SIM_CUDA_CUH
#define N_BODY_SIM_CUDA_CUH
#include <cuda_runtime.h>

void init_data(int num_particles);
void delete_data();


void call_interact_kernel(int num_blocks, int num_threads_per_block, float dt, float damping);

void get_particle_data(float3* data, float2* vel);

int get_pingpong();

int get_num_bodies();
void set_num_bodies(int n);

#endif // N_BODY_SIM_CUDA_CUH
