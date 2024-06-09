# Compiler and flags
NVCC = /usr/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 -I/usr/local/cuda/include `pkg-config --cflags opencv4`
NVCCFLAGS = -std=c++17
LDFLAGS = `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lcudart

# Target
TARGET = grayscale_conversion

# Source files
SRC = grayScale.cu
OBJ = $(SRC:.cu=.o)

# Default rule
all: $(TARGET)

# Link the target
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# Compile the source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) $(OBJ)

# # Phony targets
# .PHONY: all clean

# # Compiler and flags
# NVCC = /usr/bin/nvcc
# CXX = g++
# CXXFLAGS = -std=c++17
# NVCCFLAGS = -std=c++17
# LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

# # Target
# TARGET = minimal

# # Source files
# SRC = minimal.cu
# OBJ = $(SRC:.cu=.o)

# # Default rule
# all: $(TARGET)

# # Link the target
# $(TARGET): $(OBJ)
# 	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# # Compile the source files
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

# # Clean rule
# clean:
# 	rm -f $(TARGET) $(OBJ)

# # Phony targets
# .PHONY: all clean

# # Compiler and flags
# NVCC = /usr/bin/nvcc
# CXX = g++
# CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4`
# NVCCFLAGS = -std=c++17 -I/usr/local/cuda/include
# LDFLAGS = `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lcudart

# # Target
# TARGET = grayscalenocv

# # Source files
# SRC = grayscalenocv.cu
# OBJ = $(SRC:.cu=.o)

# # Default rule
# all: $(TARGET)

# # Link the target
# $(TARGET): $(OBJ)
# 	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# # Compile the source files
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

# # Clean rule
# clean:
# 	rm -f $(TARGET) $(OBJ)

# # Phony targets
# .PHONY: all clean

# # Compiler and flags
# NVCC = /usr/bin/nvcc
# CXX = g++
# CXXFLAGS = -std=c++17 `pkg-config --cflags opencv4`
# NVCCFLAGS = -std=c++17 -I/usr/local/cuda/include
# LDFLAGS = `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lcudart

# # Target
# TARGET = trailcv

# # Source files
# SRC = trailcv.cu
# OBJ = $(SRC:.cu=.o)

# # Default rule
# all: $(TARGET)

# # Link the target
# $(TARGET): $(OBJ)
# 	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# # Compile the source files
# %.o: %.cu
# 	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@

# # Clean rule
# clean:
# 	rm -f $(TARGET) $(OBJ)

# # Phony targets
# .PHONY: all clean
