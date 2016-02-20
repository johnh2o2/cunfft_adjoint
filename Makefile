################################################################################
#
# Build script for project
#
################################################################################
ARCH		:= sm_52

# Add source files here
EXECUTABLE      := cunfft_adjoint

################################################################################
# Rules and targets
NVCC=nvcc
CXX=g++
CC=gcc
SOURCES=

cuda_lib=/usr/local/cuda/lib64
cuda_inc=/usr/local/cuda/include

BLOCK_SIZE=256

NVCCFLAGS := -Xcompiler -fpic --ptxas-options=-v -DBLOCK_SIZE=$(BLOCK_SIZE) -arch $(ARCH)
CXXFLAGS := -fPIC -DBLOCK_SIZE=$(BLOCK_SIZE)
LINK := -lcufft -lm -lcudart -L$(cuda_lib)

all : $(EXECUTABLE)


$(EXECUTABLE): 
	$(NVCC) $(NVCCFLAGS) -o $(EXECUTABLE) $(SOURCES) $(LINK)

clean : 
	rm -f *o $(EXECUTABLE)
