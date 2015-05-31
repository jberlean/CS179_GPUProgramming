
// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cassert>
#include <fstream>
#include <iostream>

#include "n_body_sim_cuda.cuh"

void output_data(char *output_file, float *particle_data, float *particle_vels, unsigned int num_particles, float width, float height) {
  std::ofstream out(output_file);

  out << num_particles << "," << width << "," << height << std::endl;
  for (int i = 0; i < num_particles; i++) {
    out << particle_data[0] << "," << particle_data[1] << "," << particle_data[2] << std::endl;
  }

  out.close();
}

void run_simulation(
    unsigned int num_blocks,
    unsigned int threads_per_block,
    unsigned int num_particles,
    float width,
    float height,
    float total_time,
    int num_time_steps,
    int time_steps_per_frame) {
  float dt, damping;

  float *particle_data, *particle_vels;

  // Set system parameters
  dt = total_time / num_time_steps;
  damping = 0.995;

  // Allocate data structures on host
  particle_data = new float[num_particles * 3 * sizeof(float)];
  particle_vels = new float[num_particles * 2 * sizeof(float)];

  // Initialze data structures
  float v_max = std::min(width, height) / 100.0;
  init_data(num_particles, width, height, -v_max, v_max);

  // Run <time_steps> iterations of simulation
  for (int step = 0; step < num_time_steps; step++) {
    // Run kernel
    call_interact_kernel(num_blocks, threads_per_block, dt, damping);

    // Output frame data enough time steps have passed
    if (step % time_steps_per_frame == 0) {
      // Get particle data
      get_particle_data(particle_data, particle_vels);

      // Make filename
      std::string output_file;
      sprintf(output_file, "./output/data_%d.dat", step);

      // Output data
      output_data(output_file.c_str(), particle_data, particle_vels, num_particles, width, height);
    }
  }

  // Free GPU data
  delete_data();

  // Free data on host
  delete[] particle_data;
  delete[] particle_vels;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  unsigned int num_blocks, threads_per_block;

  unsigned int num_particles;
  unsigned int width, height;

  float total_time;
  unsigned int num_time_steps;

  unsigned int time_steps_per_frame;

  // Set command-line arguments
  if(argc < 9) {
      printf("Usage: n_body_sim <num-blocks> <threads-per-block> <N> <width> <height> <total-time> <num-time-steps> <time-steps-per-frame>\n");
      exit(1);
  }
  num_blocks = atoi(argv[1]);
  threads_per_block = atoi(argv[2]);
  num_particles = atoi(argv[3]);
  width = atoi(argv[4]);
  height = atoi(argv[5]);
  total_time = atof(argv[6]);
  num_time_steps = atoi(argv[7]);
  time_steps_per_frame = atoi(argv[8]);

  // make sure output directory exists
  std::ifstream test("output");
  if ((bool)test == false) {
    printf("Cannot find output directory, please make it (\"mkdir output\")\n");
    exit(1);
  }

  run_simulation(num_blocks, threads_per_block, num_particles, width, height, total_time, num_time_steps, time_steps_per_frame);
}
