// -*- C++ -*-
#ifndef N_BODY_SIM_CUDA_CUH
#define N_BODY_SIM_CUDA_CUH
#include <cuda_runtime.h>

void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block);

void delete_data();

void call_interact_kernel(float dt);

void get_particle_data(float* h_particle_data, float* h_particle_vels);

#endif // N_BODY_SIM_CUDA_CUH
