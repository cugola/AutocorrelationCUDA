NVCC=nvcc

ARCH=-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_86,code=compute_86
CUDAFLAGS=--ptxas-options=-v -m64 $(ARCH) -Xptxas -dlcm=ca -Xcompiler -D_FORCE_INLINES -lineinfo --expt-relaxed-constexpr

BIN_FOLDER=bin
SRC_FOLDER=src

FILES=${SRC_FOLDER}/main.cu ${SRC_FOLDER}/correlator.cu

all: release

clean:
	rm  -rf $(BIN_FOLDER)

release:
	mkdir -p $(BIN_FOLDER)
	$(NVCC) $(FILES) $(CUDAFLAGS) -O3 -o $(BIN_FOLDER)/main

debug:
	mkdir -p $(BIN_FOLDER)
	$(NVCC) $(FILES) $(CUDAFLAGS) -D_DEBUG_BUILD -g -G -o $(BIN_FOLDER)/main 