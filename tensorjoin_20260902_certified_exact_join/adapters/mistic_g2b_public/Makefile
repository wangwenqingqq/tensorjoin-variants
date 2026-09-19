NVCC = nvcc
CUDAFLAGS = -lcuda -Xcompiler -fopenmp -arch=compute_80 -code=sm_80 -O3
DFLAGS = 
LIBDIRS = -I.

# $(CALC_MULTI)?=4
# $(MIN_NODE_SIZE)?=1000


# DFLAGS += -DCALC_MULTI=$(CALC_MULTI)
# DFLAGS += -DMIN_NODE_SIZE=$(MIN_NODE_SIZE)
# DFLAGS += -DMAX_CALCS_PER_NODE=$(MCPN)
# DFLAGS += -DMINRP=$(RP)
# DFLAGS += -DMAXRP=$(RP)
DFLAGS += -DDIM=$(DIM)
# DFLAGS += -DKTYPE=$(KT)
DFLAGS += -DBS=$(BS)
DFLAGS += -DKB=$(KB)
# DFLAGS += -DCMP=$(CMP)
# DFLAGS += -DORDP=$(ORDP)
# DFLAGS += -DCPB=$(CPB)
# DFLAGS += -DTPP=$(TPP)
# DFLAGS += -DILP=$(ILP)

build/main: build/main.o build/launcher.o build/kernel.o build/nodes.o build/tree.o build/utils.o 
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -o build/main build/main.o build/launcher.o build/kernel.o build/nodes.o build/tree.o build/utils.o

build/main.o: src/main.cu
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -c -o build/main.o src/main.cu -lm

build/launcher.o: src/launcher.cu
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -c -o build/launcher.o src/launcher.cu -lm

build/kernel.o: src/kernel.cu
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -c -o build/kernel.o src/kernel.cu

build/nodes.o: src/nodes.cu
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -c -o build/nodes.o src/nodes.cu -lm

build/tree.o: src/tree.cu
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -c -o build/tree.o src/tree.cu -lm

build/utils.o: src/utils.cu
	$(NVCC) $(DFLAGS) $(CUDAFLAGS) $(LIBDIRS) -c -o build/utils.o src/utils.cu -lm

clean:
	rm -f build/main build/*.o
	