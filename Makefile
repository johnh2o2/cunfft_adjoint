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
#CXX=g++
CXX=$(NVCC)
CC=gcc

BLOCK_SIZE=256

ALLFLAGS := -DBLOCK_SIZE=$(BLOCK_SIZE)
NVCCFLAGS := -Xcompiler -fpic --ptxas-options=-v -arch $(ARCH)
#CXXFLAGS := -fPIC

CUDA_LIB=-L/usr/local/cuda/lib64
CUDA_INC=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc

LIBS := -lcufft -lm -lcudart

###############################################################################

CPP_FILES := $(wildcard src/*.cpp)
CU_FILES  := $(wildcard src/*.cu)

CPP_OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
CU_OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

INCLUDE_DIRS := -I./inc
INCLUDE_DIRS += $(CUDA_INC)

LIB_DIRS := $(CUDA_LIB)

LINK := $(LIBS) $(LIB_DIRS)

NVCCFLAGS += $(ALLFLAGS) $(INCLUDE_DIRS)
#CXXFLAGS += $(ALLFLAGS) $(INCLUDE_DIRS)
CXXFLAGS := $(NVCCFLAGS)
EXTRAFLAGS := 

all : $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(EXTRAFLAGS) $(CXXFLAGS) -o $@ $^ $(LINK)

$(CU_OBJ_FILES) : 
	$(NVCC) $(EXTRAFLAGS) $(NVCCFLAGS) -rdc=true -c src/$(*F).cu -o obj/$(*F).o

$(CPP_OBJ_FILES) : 
	$(CXX) $(EXTRAFLAGS) $(CXXFLAGS) -c src/$(*F).cpp -o obj/$(*F).o

.PHONY : clean
clean : 
	rm -f *dat obj/* $(EXECUTABLE)

print-%  : ; @echo $* = $($*)
