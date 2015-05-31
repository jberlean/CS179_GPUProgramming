// -*- C++ -*-
#ifndef MAIN1_CUDA_CUH
#define MAIN1_CUDA_CUH
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cuda_runtime.h>

void init_data(int num_particles);
void delete_data();

// get_force()
void interact_kernel(float4* newPos, float4* oldPos, float4* newVel, float4* oldVel, float dt, float damping, int numBodies);

int get_pingpong();

int get_num_bodies();
void set_num_bodies(int n);

#endif // MAIN1_CUDA_CUH
