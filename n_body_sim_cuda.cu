

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

unsigned int num_bodies;     // Number particles; determined at runtime.

// Device buffer variables
float2* particle_vels[2]; // x and y represent velocity in 2D
float3* particle_data[2]; // x and y represent position in 2D, z represents mass


void init_data(int num_particles, float box_width, float box_height, float min_vel, float max_vel) {
  // instantiate particle_vels, particle_data on GPU

  // set initial values for particle_vels, particle_data on GPU
}
void delete_data() {
  // free all memory on GPU
}

__device__
void get_force(/* ... */) {
  // sum force from every other particle
  // based on mass, position of both particles
    
}

__global__
void interact_kernel(float dt, float damping) {
  // each thread handles a particle

  // calculate force on particle
  // modify particle velocity
  // modify particle position
}

void call_interact_kernel(int num_blocks, int num_threads_per_block, float dt, float damping) {
  // call kernel

  // update pingpong
}


void get_particle_data(float* particle_data, float *particle_vels) {
  // copy GPU data into particle_data, particle_vels array
}


////////////////////////////////////////////////////////////////////////////////
//! Returns the value of pingpong
////////////////////////////////////////////////////////////////////////////////
int get_pingpong()
{
  return pingpong;
}

////////////////////////////////////////////////////////////////////////////////
//! Gets/sets the number of bodies
////////////////////////////////////////////////////////////////////////////////
int get_num_bodies()
{
  return num_bodies;
}
void set_num_bodies(int n)
{
  num_bodies = n;
}
