################################################################################
#
# Build script for project
#
################################################################################
ARCH		:= sm_52

# Add source files here
EXECUTABLE      := libcunfft
################################################################################
# Rules and targets
NVCC=nvcc
CC=g++

CUDA_VERSION=7.5
BLOCK_SIZE=256
VERSION=1.0

SRCDIR=src
HEADERDIR=inc
BUILDDIR=.


ifeq ($(precision), single)
  PRECISION=
else
  PRECISION=-DDOUBLE_PRECISION
endif 

ALLFLAGS := -DBLOCK_SIZE=$(BLOCK_SIZE) -DVERSION=\"$(VERSION)\" $(PRECISION)
NVCCFLAGS := $(ALLFLAGS) -Xcompiler -fpic --ptxas-options=-v -arch=$(ARCH)
CFLAGS := $(ALLFLAGS) -fPIC -Wall

CUDA_LIBS =`pkg-config --libs cudart-$(CUDA_VERSION)` 
CUDA_LIBS+=`pkg-config --libs cufft-$(CUDA_VERSION)`

CUDA_INCLUDE =`pkg-config --cflags cudart-$(CUDA_VERSION)`
CUDA_INCLUDE+=`pkg-config --cflags cufft-$(CUDA_VERSION)`

LIBS := $(CUDA_LIBS) -lm

###############################################################################

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
CU_FILES  := $(wildcard $(SRCDIR)/*.cu)

CPP_OBJ_FILES := $(notdir $(CPP_FILES:.cpp=.o))
CU_OBJ_FILES := $(notdir $(CU_FILES:.cu=.o))

INCLUDE := $(CUDA_INCLUDE) -I$(HEADERDIR)

all : $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJ_FILES) $(CU_OBJ_FILES) dlink.o
	$(CC) -shared -o $@.so $^ $(LIBS)
	mv $@.so ..

test : $(CPP_OBJ_FILES) $(CU_OBJ_FILES) dlink.o
	$(CC) -o testing $^ $(LIBS)

dlink.o : $(CU_OBJ_FILES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dlink $^ -o dlink.o

$(CU_OBJ_FILES) : 
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -rdc=true -c $(SRCDIR)/$(*F).cu -o $(*F).o

$(CPP_OBJ_FILES) : 
	$(CC) $(CFLAGS) $(INCLUDE) -c $(SRCDIR)/$(*F).cpp -o $(*F).o

.PHONY : clean
RM=rm -f


clean : 
	$(RM) *.dat *.png $(BUILDDIR)/*o testing $(EXECUTABLE).so

print-%  : ; @echo $* = $($*)
