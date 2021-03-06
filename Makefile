################################################################################
#
# Build script for project
#
################################################################################
ARCH		:= 52

# Add source files here
NAME            := cuna
################################################################################
# Rules and targets
NVCC=nvcc
CC=g++

CUDA_VERSION=7.5
BLOCK_SIZE=256
VERSION=1.3

SRCDIR=./src
HEADERDIR=./inc
BUILDDIR=./build
LIBDIR=./lib
BINDIR=./bin

CPU_OPTIMIZE=-O3
GPU_OPTIMIZE=--use_fast_math -Xcompiler -O3
#CPU_OPTIMIZE=
#GPU_OPTIMIZE=

DEFS := -DBLOCK_SIZE=$(BLOCK_SIZE) -DVERSION=\"$(VERSION)\"
NVCCFLAGS := $(DEFS) $(GPU_OPTIMIZE) -Xcompiler -fpic --ptxas-options=-v --gpu-architecture=compute_$(ARCH) --gpu-code=sm_$(ARCH),compute_$(ARCH) 
CFLAGS := $(DEFS) -fPIC -Wall $(OPTIMIZE)

CUDA_LIBS =`pkg-config --libs cudart-$(CUDA_VERSION)` 
CUDA_LIBS+=`pkg-config --libs cufft-$(CUDA_VERSION)`

CUDA_INCLUDE =`pkg-config --cflags cudart-$(CUDA_VERSION)`
CUDA_INCLUDE+=`pkg-config --cflags cufft-$(CUDA_VERSION)`

LIBS := $(CUDA_LIBS) -lm

###############################################################################

CPP_FILES := $(notdir $(wildcard $(SRCDIR)/*.cpp))
CU_FILES  := $(notdir $(wildcard $(SRCDIR)/*.cu))

CPP_OBJ_FILES_SINGLE :=$(CPP_FILES:%.cpp=$(BUILDDIR)/%f.o)
CPP_OBJ_FILES_DOUBLE :=$(CPP_FILES:%.cpp=$(BUILDDIR)/%d.o)

CU_OBJ_FILES_SINGLE := $(CU_FILES:%.cu=$(BUILDDIR)/%f.o)
CU_OBJ_FILES_DOUBLE := $(CU_FILES:%.cu=$(BUILDDIR)/%d.o)


INCLUDE := $(CUDA_INCLUDE) -I$(HEADERDIR)

all : single double test-single test-double
install : all
	sudo cp $(LIBDIR)/* /usr/local/lib
	sudo cp $(HEADERDIR)/* /usr/local/include

single : lib$(NAME)f.so
double : lib$(NAME)d.so

%f.so : $(CU_OBJ_FILES_SINGLE) $(CPP_OBJ_FILES_SINGLE) $(BUILDDIR)/dlink-single.o
	$(CC) -shared -o $(LIBDIR)/$@ $^ $(LIBS)

%d.so : $(CU_OBJ_FILES_DOUBLE) $(CPP_OBJ_FILES_DOUBLE) $(BUILDDIR)/dlink-double.o
	$(CC) -shared -o $(LIBDIR)/$@ $^ $(LIBS)

test-single :
	$(CC) $(CFLAGS) $(INCLUDE) -o $(BINDIR)/$@ test.cpp -L$(LIBDIR) -lm -l$(NAME)f

test-double : 
	$(CC) $(CFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -o $(BINDIR)/$@ test.cpp -L$(LIBDIR) -lm -l$(NAME)d

%-single.o : $(CU_OBJ_FILES_SINGLE)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dlink $^ -o $@

%-double.o : $(CU_OBJ_FILES_DOUBLE)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -dlink $^ -o $@

$(CU_OBJ_FILES_SINGLE) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -rdc=true -c $(SRCDIR)/$(notdir $(subst f.o,.cu,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CU_OBJ_FILES_DOUBLE) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -rdc=true -c $(SRCDIR)/$(notdir $(subst d.o,.cu,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CPP_OBJ_FILES_SINGLE) : 
	$(CC) $(CFLAGS) $(INCLUDE) -c $(SRCDIR)/$(notdir $(subst f.o,.cpp,$@)) -o $(BUILDDIR)/$(notdir $@)

$(CPP_OBJ_FILES_DOUBLE) : 
	$(CC) $(CFLAGS) $(INCLUDE) -DDOUBLE_PRECISION -c $(SRCDIR)/$(notdir $(subst d.o,.cpp,$@)) -o $(BUILDDIR)/$(notdir $@)

.PHONY : clean
RM=rm -f


clean : 
	$(RM) *.dat *.png $(BUILDDIR)/* $(BINDIR)/* $(LIBDIR)/* 

print-%  : ; @echo $* = $($*)
