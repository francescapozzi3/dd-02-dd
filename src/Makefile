# From inside of src/:
#
# COMPILE: make
#
# RUN: make run
#  Personalize number of processors: make run NPROCS=4
#  Personalize parameters:           make run NX=200 NY=200 NPROCS=16 OVERLAP=3
#
# RUN SEQUENTIALLY: make run_seq
#
# CLEAN: 
#  Remove only .o and executable: make clean
#  Remove build/ and output/:     make distclean


# Compiler and flags
CXX = mpicxx
CXXFLAGS = -O3 -std=c++17 -Wall
INCLUDES = -I../include -I/usr/include/eigen3

# Directories
SRC_DIR = .
BUILD_DIR = ../build
OUTPUT_DIR = ../output
INC_DIR = ../include

# Source files
SOURCES = main.cpp ras_2d.cpp
OBJECTS = $(addprefix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))

# Executable
TARGET = $(BUILD_DIR)/ras_2d_mpi

# Default number of processors
NPROCS ?= 8

# Grid size and other parameters (can be overridden)
NX      ?= 100
NY      ?= 100
LX      ?= 1.0
LY      ?= 1.0
MU      ?= 1.0
C       ?= 0.1
OVERLAP ?= 2
MAX_IT  ?= 2000
TOL     ?= 1e-06
RESTART ?= 50

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(OUTPUT_DIR)

# Link executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run with multiple processors
run: $(TARGET)
	@echo "Running with $(NPROCS) processors..."
	@cd $(OUTPUT_DIR) && mpirun -np $(NPROCS) $(TARGET) $(NX) $(NY) $(LX) $(LY) $(MU) $(C) $(OVERLAP) $(MAX_IT) $(TOL) $(RESTART)

# Run sequentially (1 processor)
run_seq: $(TARGET)
	@echo "Running sequentially (1 processor)..."
	@cd $(OUTPUT_DIR) && mpirun -np 1 $(TARGET) $(NX) $(NY) $(LX) $(LY) $(MU) $(C) $(OVERLAP) $(MAX_IT) $(TOL) $(RESTART) 

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

# Clean everything including output
distclean: clean
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR)

# Phony targets
.PHONY: all directories run run_seq clean distclean

# Dependencies (header files)
$(BUILD_DIR)/main.o: $(INC_DIR)/ras_2d.hpp
$(BUILD_DIR)/ras_2d.o: $(INC_DIR)/ras_2d.hpp $(INC_DIR)/gmres.hpp $(INC_DIR)/bicgstab.hpp \
                       $(INC_DIR)/gmres_util.hpp $(INC_DIR)/LinearAlgebraTraits.hpp
