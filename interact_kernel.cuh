// -*- C++ -*-
#ifndef MAIN1_CUDA_CUH
#define MAIN1_CUDA_CUH
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cuda_runtime.h>

void interact_kernel(float4* newPos, float4* oldPos, float4* newVel, float4* oldVel, float dt, float damping, int numBodies);
void runCuda(GLuint *vbo, float dt);
void createDeviceData();
void createVBOs(GLuint* vbo);
void deleteVBOs(GLuint* vbo);
void deleteDeviceData();
int getPingpong();
int getNumBodies();
void setNumBodies(int n);

#endif // MAIN1_CUDA_CUH
