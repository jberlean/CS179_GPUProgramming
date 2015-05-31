### CHANGE THESE LINES TO MATCH YOUR SYSTEM        ###
### COMPILER PATH                                  ###
CC = /usr/bin/g++
### CUDA FOLDER PATH                               ###
CUDA_PATH       ?= /usr/local/cuda-6.5
# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30
# library flags -- on linux, this may look like -lgl -lglut
LD_FLAGS = -framework OpenGL -framework GLUT
######################################################

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

CC_INCLUDE = $(CUDA_PATH)/samples/common/inc

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = n_body_sim

all: $(TARGETS)

n_body_sim: n_body_sim.cc n_body_sim.o
	$(CC) $< -o $@ n_body_sim.o -O3 -I$(CUDA_INC_PATH) $(LDFLAGS) $(LD_FLAGS) -Wall

n_body_sim.o: n_body_sim_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -I$(CC_INCLUDE) -o $@ -c $<


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
