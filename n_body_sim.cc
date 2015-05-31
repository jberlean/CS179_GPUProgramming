
// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cassert>

#include "n_body_sim_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  if(argc < 2)
  {
      printf("Usage: ./n_body_sim <N> <width> <height> <total_time> <time_steps>\n");
      exit(1);
  }
  // set cmd-line args + system parameters

  // initialze data structures

  // run <time_steps> iterations of simulation
  // upload particle data to GPU
  // for each iteration
      // run kernel
      // copy results from GPU + store in file at given intervals

  // delete data

}
