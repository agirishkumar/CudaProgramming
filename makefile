# Compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include `pkg-config --cflags opencv4`
NVCCFLAGS = -std=c++17
LDFLAGS = `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lcudart

# Targets
TARGETS = scanAlgorithm grayscale_conversion

# Source files for each target
SCAN_SOURCES = scanAlgorithm.cu 
SCAN_OBJECTS = $(SCAN_SOURCES:.cu=.o)

GRAYSCALE_SOURCES = grayScale.cu 
GRAYSCALE_OBJECTS = $(GRAYSCALE_SOURCES:.cu=.o)

# Default rule
all: $(TARGETS)

# Linking rules for each target
scanAlgorithm: $(SCAN_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

grayscale_conversion: $(GRAYSCALE_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compile the source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGETS) $(SCAN_OBJECTS) $(GRAYSCALE_OBJECTS)

# Phony targets
.PHONY: all clean
