#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

#include "n_body_sim_cuda.cuh"

void output_data_header(std::ofstream &out, int num_particles, float width, float height, float total_time, int num_time_steps, int time_steps_per_frame, std::string algorithm) {
  out << num_particles << " "
      << width << " "
      << height << " "
      << total_time << " "
      << num_time_steps << " "
      << time_steps_per_frame << " " 
      << algorithm << std::endl;

}

void output_data(std::ofstream &out, float *particle_data, float *particle_vels, int frame_num, int num_particles, float width, float height) {
  out << frame_num << std::endl;
  for (int i = 0; i < num_particles; i++) {
    out << particle_data[i*3] << " " << particle_data[i*3 + 1] << " " << particle_data[i*3 + 2] << " " << particle_vels[i*2] << " " << particle_vels[i*2 + 1] << std::endl;
  }
}

void load_input_file(char *infile,
    int &num_blocks, int &num_threads_per_block,
    int &num_particles, float &width, float &height,
    float &total_time, int &num_time_steps, int &time_steps_per_frame) {
  std::ifstream in(infile);
  float *particle_data, *particle_vels;

  in >> num_blocks >> num_threads_per_block >> num_particles >> width >> height
      >> total_time >> num_time_steps >> time_steps_per_frame;

  particle_data = new float[3 * num_particles];
  particle_vels = new float[2 * num_particles];

  for (int i = 0; i < num_particles; i++) {
    in >> particle_data[3*i] >> particle_data[3*i + 1] >> particle_data[3*i + 2]
        >> particle_vels[2*i] >> particle_vels[2*i + 1];
  }

  init_data(num_particles, particle_data, particle_vels, num_blocks, num_threads_per_block);

  delete[] particle_data;
  delete[] particle_vels;
}

void run_simulation(
    int num_blocks,
    int num_threads_per_block,
    int num_particles,
    float width,
    float height,
    float total_time,
    int num_time_steps,
    int time_steps_per_frame) {
  float dt;
  std::ofstream out;

  float *particle_data, *particle_vels;

  // Setup output stuff (filename and output stream)
  char output_file[200];
  sprintf(output_file, "./output/data_%d.dat", (int)time(NULL));
  out.open(output_file);

  // Set system parameters
  dt = total_time / num_time_steps;

  // Allocate data structures on host
  particle_data = new float[num_particles * 3];
  particle_vels = new float[num_particles * 2];

  // Output header for data file
  output_data_header(out, num_particles, width, height, total_time, num_time_steps, time_steps_per_frame, get_algorithm());

  // Run <time_steps> iterations of simulation
  long status_counter = 0;
  for (int step = 0; step < num_time_steps; step++) {
    // Update the particle data after <dt> time
    simulate_time_step(dt);

    status_counter += num_particles*num_particles;
    if (status_counter > 500000000) {
      std::cout << "Run " << step << " time steps\n";
      status_counter = 0;
    }

    // Output frame data enough time steps have passed
    if (step % time_steps_per_frame == 0) {
      // Get particle data
      get_particle_data(particle_data, particle_vels);

      // Output data
      output_data(out, particle_data, particle_vels, step/time_steps_per_frame, num_particles, width, height);
    }
  }

  // Close output stream (all output finished)
  out.close();
  std::cout << "Output data to " << output_file << std::endl;

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
  int num_blocks, num_threads_per_block;
  int num_particles;
  float width, height;

  float total_time;
  int num_time_steps;

  int time_steps_per_frame;

  // Set command-line arguments
  if (argc == 2) {
    load_input_file(argv[1], num_blocks, num_threads_per_block, num_particles, width, height, total_time, num_time_steps, time_steps_per_frame);
  } else if(argc == 3 || argc == 9) {
      if (argc == 3) {
        width = 512;
        height = 512;
        total_time = 10;
        num_time_steps = 1000;
        time_steps_per_frame = 10;
      } else if (argc == 9) {
        num_blocks = atoi(argv[1]);
        num_threads_per_block = atoi(argv[2]);
        num_particles = atoi(argv[3]);
        width = atof(argv[4]);
        height = atof(argv[5]);
        total_time = atof(argv[6]);
        num_time_steps = atoi(argv[7]);
        time_steps_per_frame = atoi(argv[8]);
      }
      num_blocks = atoi(argv[1]);
      num_threads_per_block = atoi(argv[2]);
      num_particles = atoi(argv[3]);

      float v_max = std::min(width, height) / 1000.0;
      init_data(num_particles, width, height, -v_max, v_max, num_blocks, num_threads_per_block);
  } else {
      printf("Usage: %s <num-blocks> <num-threads-per-block> <N> [<width> <height> <total-time> <num-time-steps> <time-steps-per-frame>]\n", argv[0]);
      exit(1);
  }
    
  // Make sure output directory exists
  std::ifstream test("output");
  if ((bool)test == false) {
    printf("Cannot find output directory, please make it (\"mkdir output\")\n");
    exit(1);
  }

  std::cout << "Initialization complete. Beginning simulation.\n";

  // Run simulation with given parameters
  run_simulation(num_blocks, num_threads_per_block, num_particles, width, height, total_time, num_time_steps, 
                 time_steps_per_frame);
}
