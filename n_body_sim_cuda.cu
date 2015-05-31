#include <curand.h>

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

// Flag for pingpong;
int pingpong = 0;

int num_particles;     // Number particles; determined at runtime.
int num_blocks;
int num_threads_per_block;

// Device buffer variables
float2* particle_vels[2]; // x and y represent velocity in 2D
float3* particle_data[2]; // x and y represent position in 2D, z represents mass


void init_data(int h_num_particles, float box_width, float box_height, float min_vel, 
               float max_vel, int h_num_blocks, int h_num_threads_per_block) 
{
  num_particles = h_num_particles;
  num_blocks = h_num_blocks;
  num_threads_per_block = h_num_threads_per_block;

  // instantiate particle_vels, particle_data on GPU
  gpuErrChk(cudaMalloc((void **) &particle_vels[0], sizeof(float2) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_vels[1], sizeof(float2) * num_particles));
  
  gpuErrChk(cudaMalloc((void **) &particle_data[0], sizeof(float3) * num_particles));
  gpuErrChk(cudaMalloc((void **) &particle_data[1], sizeof(float3) * num_particles));
   
  // set initial values for particle_vels, particle_data on GPU
  float * random;
  gpuErrChk(cudaMalloc((void **) &random, sizeof(float) * num_particles * 4);   
  
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateUniform(gen, random, num_particles * 4);

  cudaInitKernel<<<num_blocks, num_threads_per_block>>>(random, box_width, box_height, min_vel, max_vel);

  curandDestroyGenerator(gen);
  gpuErrChk(cudaFree(random));
}

__global__
void cudaInitKernel(float * random, float box_width, float box_height, float min_vel, float max_vel)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < num_particles)
  {
    particle_vels[0][i].x = min_vel + random[4 * i] * (max_vel - min_vel);
    particle_vels[0][i].y = min_vel + random[4 * i + 1] * (max_vel - min_vel);
    particle_data[0][i].x = random[4 * i + 2] * box_width;
    particle_data[0][i].y = random[4 * i + 3] * box_height;
    particle_data[0][i].z = 1;

    particle_data[1][i].z = 1;    

    i += blockDim.x * gridDim.x;
  }
}

void delete_data() {
  // free all memory on GPU
  for (int i = 0; i < 2; i++)
  {
    gpuErrChl(cudaFree(particle_vels[i]));
    gpuErrChl(cudaFree(particle_data[i]));
  }
}

__device__
float2 get_force(int pos) {
  // sum force from every other particle
  // based on mass, position of both particles
  float2 force;
  force.x = 0;
  force.y = 0;

  for (int i = 0; i < num_particles; i++)
  {
    float dist_squared = pow((particle_data[pingpong][pos].x - particle_data[pingpong][i].x), 2) 
                         + pow((particle_data[pingpong][pos].y - particle_data[pingpong][i].y), 2);  

    if (dist_squared > 0)
    {
      float force_magnitude = particle_data[pingpong][pos].z * particle_data[pingppong][i].z / dist_squared;
      force.x += (particle_data[pingpong][i].x - particle_data[pingpong][pos].x) * force_magnitude / sqrt(dist_squared);
      force.y += (particle_data[pingpong][i].y - particle_data[pingpong][pos].y) * force_magnitude / sqrt(dist_squared);
    }
  }
  return force;  
}

__global__
void interact_kernel(float dt) {
  // each thread handles a particle
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < num_bodies)
  {
    float2 force = get_force(i);
    
    particle_vels[1 - pingpong][i].x += force.x * dt / particle_data[0][i].z;
    particle_vels[1 - pingpong][i].y += force.y * dt / particle_data[0][i].z;
    
    particle_data[1 - pingpong][i].x += particle_vels[1 - pingpong][i].x * dt; 
    particle_data[1 - pingpong][i].y += particle_vels[1 - pingpong][i].y * dt;

    i += blockDim.x * gridDim.x;
  }
}

void call_interact_kernel(float dt) {
  // call kernel
  interact_kernel<<<num_blocks, num_threads_per_block>>>(dt);
  // update pingpong
  pingpong = 1 - pingpong;
}


void get_particle_data(float * h_particle_data, float * h_particle_vels) {
  // copy GPU data into particle_data, particle_vels array
  gpuErrChk(cudaMemcpy(h_particle_data, particle_data[1 - pingpong], sizeof(float) * 3 * num_particles, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(h_particle_vels, particle_vels[1 - pingpong], sizeof(float) * 2 * num_particles, cudaMemcpyDeviceToHost));
}
