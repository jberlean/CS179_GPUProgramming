CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcurand
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcurand
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcurand
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64 --ptxas-options="-v"
endif


TARGETS = sim_cpu sim_simple sim_pxp sim_pxp_opt sim_simple_coalesced sim_pxp_coalesced sim_pxp_opt_coalesced

all: $(TARGETS)

sim_cpu: n_body_sim.cc n_body_sim_cpu.o
	$(CC) $< -o $@ n_body_sim_cpu.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
n_body_sim_cpu.o: n_body_sim_cpu.cc
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

sim_simple: n_body_sim.cc cuda_simple.o
	$(CC) $< -o $@ cuda_simple.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
cuda_simple.o: cuda_simple.cu cuda_general_noncoal.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

sim_pxp: n_body_sim.cc cuda_pxp.o
	$(CC) $< -o $@ cuda_pxp.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
cuda_pxp.o: cuda_pxp.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

sim_pxp_opt: n_body_sim.cc cuda_pxp_opt.o
	$(CC) $< -o $@ cuda_pxp_opt.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
cuda_pxp_opt.o: cuda_pxp_opt.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

sim_simple_coalesced: n_body_sim.cc cuda_simple_coal.o
	$(CC) $< -o $@ cuda_simple_coal.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
cuda_simple_coal.o: cuda_simple_coal.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

sim_pxp_coalesced: n_body_sim.cc cuda_pxp_coal.o
	$(CC) $< -o $@ cuda_pxp_coal.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
cuda_pxp_coal.o: cuda_pxp_coal.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

sim_pxp_opt_coalesced: n_body_sim.cc cuda_pxp_opt_coal.o
	$(CC) $< -o $@ cuda_pxp_opt_coal.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -fopenmp
cuda_pxp_opt_coal.o: cuda_pxp_opt_coal.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
