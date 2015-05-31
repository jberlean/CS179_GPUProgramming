#include <helper_math.h>
#include <stdio.h>

#include <cuda_gl_interop.h>
////////////////////////////////////////////////////////////////////////////////
// constants & defines
//TODO:: Choose one!
// Number of threads in a block.
#define BLOCK_SIZE 512

// macro for error-handling
#define gpuErrchk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define WRAP(x,m) ((x)<(m)?(x):((x)-(m)))
// The maximum radius of a neighborhood
#define MAX_DIST 20

// Flag for pingpong;
int pingpong = 0;

unsigned int numBodies;     // Number particles; determined at runtime.

// Device buffer variables
float2* particle_vels[2]; // x and y represent velocity in 2D
float3* particle_data[2]; // x and y represent position in 2D, z represents mass

__device__ float4 get_force(float4 pos, float4* neighbors)
{
    float4 accel = make_float4(0.0, 0.0, 0.0, 0.0);
    float4 avg_heading = make_float4(0.0, 0.0, 0.0, 0.0);
    int visible_neighbor_count = 0;
    // Iterate through to get cohesion force first
    for (int i = 0; i < blockDim.x; i++)
    {
        int index = WRAP(threadIdx.x + i, blockDim.x);
        float4 neighbor = neighbors[index];
        float dist = sqrt(dot(pos - neighbor, pos - neighbor));
        // If the flock is in our neighborhood, increment the neighbor count
        // and pull the acceleration toward the neighbor
        if (dist < MAX_DIST)
        {
            visible_neighbor_count++;
            accel += neighbor - pos;
        }
    }
    // If there's no neighbors, there's no acceleration...
    if (visible_neighbor_count == 0) return accel;
    // Average out the acceleration
    accel /= visible_neighbor_count;

    // Now we iterate through again to get separation force
    for (int i = 0; i < blockDim.x; i++)
    {
        int index = WRAP(threadIdx.x + i, blockDim.x);
        float4 neighbor = neighbors[index];
        float dist = sqrt(dot(pos - neighbor, pos - neighbor));
        // If its in our neighborhood, apply the force (inverse squared idea)
        if (dist != 0 && dist < MAX_DIST)
            accel += 0.05 * (pos - neighbor) / (dist * dist * dist);
    }
    return 100 * accel;
}
__global__ void interact_kernel(float4* newPos, float4* oldPos, float4* newVel, float4* oldVel, float dt, float damping, int numBodies)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ float4 pos[];
    if (x < numBodies)
      pos[threadIdx.x] = oldPos[x];
    syncthreads();
    
    if (x >= numBodies) return;
    // Calculate the new velocity
    newVel[x] = oldVel[x];
    newVel[x] += get_force(oldPos[x], pos) * dt;
    // Normalize it to constant speed
    newVel[x] = 100.0 * normalize(newVel[x]);
    
    // Update position
    newPos[x] = oldPos[x] + newVel[x] * dt;
    newPos[x].w = 1.0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(GLuint *vbo, float dt)
{
    // map OpenGL buffer object for writing from CUDA
    float4* oldPos;
    float4* newPos;

    // Velocity damping factor
    float damping = 0.995;


    // TODO:: Map opengl buffers to CUDA.
    cudaGLMapBufferObject((void**)&oldPos, vbo[pingpong]);
    cudaGLMapBufferObject((void**)&newPos, vbo[1 - pingpong]);

    // TODO:: Choose a block size, a grid size, an amount of shared mem,
    // and execute the kernel
    // dVels is the particle velocities old, new. Pingponging of these is
    // handled, if the initial conditions have initial velocities in dVels[0].
    dim3 block(min(BLOCK_SIZE, numBodies), 1, 1);
    dim3 grid(ceil(numBodies / BLOCK_SIZE), 1, 1);
    // Assumes 48kb max shared memory
    size_t sharedMemSize = min(numBodies, 3072) * sizeof(float4);
    interact_kernel<<<grid, block, sharedMemSize>>>
      (newPos, oldPos, dVels[1 - pingpong], dVels[pingpong], 
      dt, damping, numBodies);

    // TODO:: unmap buffer objects from cuda.
    cudaGLUnmapBufferObject(vbo[0]);
    cudaGLUnmapBufferObject(vbo[1]);

    // TODO:: Switch buffers between old/new
    pingpong ^= 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Create device data
////////////////////////////////////////////////////////////////////////////////
void createDeviceData()
{
    gpuErrchk(cudaMalloc((void**)&dVels[0], numBodies *
                                                    4 * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dVels[1], numBodies *
                                                    4 * sizeof(float)));

    // Initialize data.
    float4* tempvels = (float4*)malloc(numBodies * 4*sizeof(float));
    for(int i = 0; i < numBodies; ++i)
    {
        // TODO: set initial velocity data
        tempvels[i].x = (2 * (float)rand() / RAND_MAX - 1);
        tempvels[i].y = (2 * (float)rand() / RAND_MAX - 1);
        tempvels[i].z = (2 * (float)rand() / RAND_MAX - 1);
        tempvels[i].w = 1.f;
    }

    // Copy to gpu
    gpuErrchk(cudaMemcpy(dVels[0], tempvels, numBodies*4*sizeof(float), cudaMemcpyHostToDevice));

    free(tempvels);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBOs(GLuint* vbo)
{
/*    glewInit();

    // create buffer object
    glGenBuffers(2, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
*/
    // initialize buffer object; this will be used as 'oldPos' initially
    unsigned int size = numBodies * 4 * sizeof(float); //change to 2D, size = numBodies * 2 * sizeof(float)

    float4* temppos = (float4*)malloc(numBodies*4*sizeof(float)); // change to 2D
    for(int i = 0; i < numBodies; ++i)
    {/*
      // Added in some fancy math to make them start out similar to the
      // demo video on the website.
      float rand_theta = ((float)rand()) / RAND_MAX * 2.0 * M_PI;
      float rand_phi = ((float)rand()) / RAND_MAX * 2.0 * M_PI - M_PI;
      float rand_radius = ((float)rand()) / RAND_MAX * 3.0;
      temppos[i].x = -8.0 + (i % 2) * 16.0 + rand_radius * cos(rand_theta) * cos(rand_phi);
      temppos[i].y = -8.0 + ((i / 2) % 2) * 16.0 + rand_radius * sin(rand_theta) * cos(rand_phi);
      temppos[i].z = rand_radius * sin(rand_phi);
      temppos[i].w = 1.;*/
      // change to 2D
    }

    // Notice only vbo[0] has initial data!
    //glBufferData(GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    free(temppos);

    // Create initial 'newPos' buffer
    /*glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer objects with CUDA
    gpuErrchk(cudaGLRegisterBufferObject(vbo[0]));
    gpuErrchk(cudaGLRegisterBufferObject(vbo[1]));*/
}
/*
////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBOs(GLuint* vbo)
{
    glBindBuffer(1, vbo[0]);
    glDeleteBuffers(1, &vbo[0]);
    glBindBuffer(1, vbo[1]);
    glDeleteBuffers(1, &vbo[1]);

    gpuErrchk(cudaGLUnregisterBufferObject(vbo[0]));
    gpuErrchk(cudaGLUnregisterBufferObject(vbo[1]));

    *vbo = 0;
}*/

////////////////////////////////////////////////////////////////////////////////
//! Delete device data
////////////////////////////////////////////////////////////////////////////////
void deleteDeviceData()
{
    // Create a velocity for every position.
    gpuErrchk(cudaFree(dVels[0]));
    gpuErrchk(cudaFree(dVels[1]));
    // pos's are the VBOs
}

////////////////////////////////////////////////////////////////////////////////
//! Returns the value of pingpong
////////////////////////////////////////////////////////////////////////////////
int getPingpong()
{
  return pingpong;
}

////////////////////////////////////////////////////////////////////////////////
//! Gets/sets the number of bodies
////////////////////////////////////////////////////////////////////////////////
int getNumBodies()
{
  return numBodies;
}
void setNumBodies(int n)
{
  numBodies = n;
}
